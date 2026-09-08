#!/usr/bin/env python
"""Does seed-averaging the GNN leg beat the single-seed r1-10 blend (2.215)?

The shipped champion GNN (gnn_attentive_10rounds_v1) is ONE seed. This loads that
plus any of gnn_attentive_10rounds_seed{1..4} that exist, runs each checkpoint on
the frozen scaffold-disjoint holdout, averages the GNN predictions over a growing
number of seeds, and re-sweeps the blend weight against the shipped MLP+XGB
ensemble each time.

  /home/schen3/venv/nequip/bin/python cross_benzoin/blend_gnn_seed_ensemble_r10.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "cross_benzoin"))
from train_cross_delta import TARGET_COL, BASELINE_COL  # noqa: E402
from train_cross_ensemble import MLPXGBEnsemble  # noqa: E402,F401
from train_cross_gnn import build_graphs, make_loader, evl  # noqa: E402
from gnn_architectures import TripleGNNAttn  # noqa: E402

R10 = REPO / "data/cross_benzoin/cross_round10"
TABLE = R10 / "cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
ENS_PATH = R10 / "scaffold_disjoint_10rounds_v1/models/ensemble_scaffold_disjoint.joblib"
SEED_DIRS = [R10 / "gnn_attentive_10rounds_v1"] + [R10 / f"gnn_attentive_10rounds_seed{s}" for s in (1, 2, 3, 4)]


def gnn_holdout_preds(seed_dir: Path, df: pd.DataFrame):
    ns = joblib.load(seed_dir / "models" / "gnn_norm_stats.joblib")
    feats = ns["feats"]
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    Xz = Xdf.fillna(ns["med"]).fillna(0.0)
    qmz = ((Xz.to_numpy() - ns["qm_mean"]) / ns["qm_std"]).astype(np.float32)
    pairs = build_graphs(df, qmz)
    ym, ysd = ns["ym"], ns["ysd"]
    for p in pairs:
        p.ystd = (p.y - ym) / ysd
    id_to_split = dict(zip(df["id"].astype(str), df["_split"]))
    te = [p for p in pairs if id_to_split.get(p.row_id) == "test"]
    ad, bd = pairs[0].x_p.shape[1], pairs[0].edge_attr_p.shape[1]
    model = TripleGNNAttn(ad, bd, len(feats), h=ns["hidden"], layers=ns["layers"])
    model.load_state_dict(torch.load(seed_dir / "models" / "gnn_state.pt", map_location="cpu"))
    from train_cross_gnn import dev
    model = model.to(dev)
    mae, yh, yt, ids = evl(model, make_loader(te, 64, False), ym, ysd)
    return dict(zip(ids, yh)), dict(zip(ids, yt)), mae


def main() -> int:
    df = pd.read_parquet(TABLE)
    # mirror arch_sweep's split column
    if "_split" not in df.columns:
        col = "new_scaffold_split" if "new_scaffold_split" in df.columns else "scaffold_split"
        df["_split"] = df[col]
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)

    have = [d for d in SEED_DIRS if (d / "models" / "gnn_state.pt").exists()]
    print(f"seed checkpoints found: {[d.name for d in have]}")
    if not have:
        print("no GNN checkpoints yet")
        return 1

    per_seed_pred, true_by_id, per_seed_mae = [], None, []
    for d in have:
        gp, tp, mae = gnn_holdout_preds(d, df)
        per_seed_pred.append(gp)
        per_seed_mae.append(mae)
        true_by_id = tp
        print(f"  {d.name:32s} GNN-only holdout MAE {mae:.4f}")

    ens = joblib.load(ENS_PATH)
    tdf = df[df["_split"] == "test"].copy()
    tdf.index = tdf["id"].astype(str)
    ens_pred_by_id = dict(zip(tdf.index, ens.predict(tdf) + tdf[BASELINE_COL].to_numpy()))

    ids = [i for i in per_seed_pred[0] if i in ens_pred_by_id and i in true_by_id]
    y = np.array([true_by_id[i] for i in ids])
    ens_p = np.array([ens_pred_by_id[i] for i in ids])
    mae_ens = float(np.abs(y - ens_p).mean())
    print(f"\nensemble-only holdout MAE (n={len(ids)}): {mae_ens:.4f}")

    print(f"\n{'#seeds':>7} {'gnn_ens_MAE':>12} {'best_w_gnn':>11} {'blend_MAE':>10} {'vs 2.215':>9}")
    rows = []
    for k in range(1, len(per_seed_pred) + 1):
        gnn_p = np.mean([[per_seed_pred[s][i] for i in ids] for s in range(k)], axis=0)
        mae_gnn = float(np.abs(y - gnn_p).mean())
        best_w, best_mae = 0.0, mae_ens
        for w in np.arange(0.05, 1.0, 0.05):
            m = float(np.abs(y - ((1 - w) * ens_p + w * gnn_p)).mean())
            if m < best_mae:
                best_mae, best_w = m, float(w)
        print(f"{k:>7} {mae_gnn:>12.4f} {best_w:>11.2f} {best_mae:>10.4f} {best_mae-2.215:>+9.4f}")
        rows.append({"n_seeds": k, "gnn_ens_only_mae": mae_gnn, "best_w_gnn": best_w,
                     "blend_mae": best_mae, "delta_vs_shipped_2215": best_mae - 2.215})

    out = R10 / "gnn_seed_ensemble_r10_result.json"
    out.write_text(json.dumps({"n_holdout": len(ids), "ensemble_only_mae": mae_ens,
                               "per_seed_gnn_only_mae": per_seed_mae,
                               "seed_dirs": [d.name for d in have], "sweep": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
