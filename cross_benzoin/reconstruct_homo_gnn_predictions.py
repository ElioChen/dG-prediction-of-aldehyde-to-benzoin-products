#!/usr/bin/env python
"""Reload trained homo-full-library GNN checkpoint(s) and dump per-row test
predictions to CSV, for the model-benchmark notebook (2026-09-20) to plot --
`train_cross_gnn_arch_sweep.py` itself only saves summary metrics + the
checkpoint, not a predictions table, and the notebook's own kernel
(nhc-workflow) has no torch, so this precompute step needs the nequip env.

Usage:
    /home/schen3/venv/nequip/bin/python cross_benzoin/reconstruct_homo_gnn_predictions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os  # noqa: E402
os.environ.setdefault("CB_BASELINE_COL", "dG_b973c_kcal")

from train_cross_gnn import build_graphs, make_loader, evl, TripleGNN  # noqa: E402
from gnn_architectures import TripleGNNAttn  # noqa: E402
from train_cross_gnn_arch_sweep import MODEL_CLASSES  # noqa: E402
import torch  # noqa: E402

TABLE = REPO / "data/cross_benzoin/homo_standalone/homo_standalone_full_library_table_slim260.parquet"
OUT = REPO / "data/cross_benzoin/homo_standalone/gnn_benchmark_predictions.csv"

RUNS = {
    "attentive": REPO / "data/cross_benzoin/homo_standalone/gnn_attentive_full_seed0",
    "default": REPO / "data/cross_benzoin/homo_standalone/gnn_default_full_seed0",
}


def reconstruct_one(run_dir: Path, df: pd.DataFrame) -> pd.DataFrame | None:
    stats_path = run_dir / "models" / "gnn_norm_stats.joblib"
    state_path = run_dir / "models" / "gnn_state.pt"
    if not (stats_path.exists() and state_path.exists()):
        print(f"skip {run_dir.name}: no saved checkpoint yet")
        return None
    stats = joblib.load(stats_path)
    feats = stats["feats"]

    d = df[df["new_scaffold_split"] != "mixed"].reset_index(drop=True)
    d["_split"] = d["new_scaffold_split"]
    test_mask = (d["_split"] == "test").to_numpy()

    Xdf = d[feats].apply(pd.to_numeric, errors="coerce")
    Xz = Xdf.fillna(stats["med"]).fillna(0.0)
    qmz = ((Xz.to_numpy() - stats["qm_mean"]) / stats["qm_std"]).astype(np.float32)

    pairs = build_graphs(d, qmz)
    id_to_split = dict(zip(d["id"].astype(str), d["_split"]))
    te_pairs = [p for p in pairs if id_to_split.get(p.row_id) == "test"]
    for p in te_pairs:
        p.ystd = (p.y - stats["ym"]) / stats["ysd"]

    ModelCls = MODEL_CLASSES[stats["arch"]]
    model = ModelCls(stats["ad"], stats["bd"], stats["nqm"], h=stats["hidden"], layers=stats["layers"])
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.to("cpu")

    from train_cross_gnn import dev as _dev_module  # noqa: F401
    import train_cross_gnn as tcg
    tcg.dev = torch.device("cpu")  # force CPU for this reload, avoids needing a GPU node just to read predictions

    mae, yh, yt, ids_ = evl(model, make_loader(te_pairs, 64, False), stats["ym"], stats["ysd"])
    print(f"{run_dir.name}: reconstructed test MAE={mae:.4f} (n={len(ids_)})")
    return pd.DataFrame({"id": ids_, "dG_true": yt, f"pred_gnn_{run_dir.name}": yh})


def main() -> int:
    df = pd.read_parquet(TABLE)
    merged = None
    for name, run_dir in RUNS.items():
        out = reconstruct_one(run_dir, df)
        if out is None:
            continue
        if merged is None:
            merged = out
        else:
            merged = merged.merge(out.drop(columns=["dG_true"]), on="id", how="outer")
    if merged is None:
        print("nothing reconstructed")
        return 1
    merged.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(merged)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
