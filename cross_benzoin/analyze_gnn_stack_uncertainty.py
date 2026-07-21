#!/usr/bin/env python
"""Deeper look at the GNN+ensemble frozen-holdout result (n=29) before trusting it:
per-row error breakdown + bootstrap CIs on the MAE deltas, since n=29 with a huge
dG_orca range (-35 to +28 kcal/mol seen in this exact test set) is exactly the
regime where 1-2 outlier rows can dominate a raw MAE comparison.

Reloads the already-trained artifacts (ensemble joblib, GNN state dict) -- no
retraining, pure inference + analysis.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cross_delta import TARGET_COL, BASELINE_COL, pair_split_labels  # noqa: E402
from train_cross_ensemble import MLPXGBEnsemble  # noqa: E402,F401
from train_cross_gnn import (  # noqa: E402
    build_graphs, make_loader, evl, TripleGNN,
)
import __main__ as _main_mod  # noqa: E402
_main_mod.MLPXGBEnsemble = MLPXGBEnsemble

DEFAULT_TABLE = REPO / "data/cross_benzoin/cross_round5/cross_train_table_5rounds_mordred_slim120.parquet"
DEFAULT_CHAMPION_DIR = REPO / "data/cross_benzoin/cross_round5/train_5rounds_mordred_slim120_v1"
DEFAULT_ENSEMBLE_DIR = REPO / "data/cross_benzoin/cross_round5/train_ensemble_slim120_v1"
DEFAULT_GNN_DIR = REPO / "data/cross_benzoin/cross_round5/train_gnn_transfer_v1"
DEFAULT_OUT = REPO / "data/cross_benzoin/cross_round5/gnn_stack_uncertainty"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--champion-dir", type=Path, default=DEFAULT_CHAMPION_DIR)
    ap.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    ap.add_argument("--gnn-dir", type=Path, default=DEFAULT_GNN_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    TABLE, CHAMPION_DIR, ENSEMBLE_DIR, GNN_DIR, OUT = (
        args.table, args.champion_dir, args.ensemble_dir, args.gnn_dir, args.out)

    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")

    df = pd.read_parquet(TABLE)
    feats = json.loads((CHAMPION_DIR / "models" / "feature_list.json").read_text())
    pair_split = pair_split_labels(df, verbose=False)
    df = df.assign(_split=pair_split)
    df["_split"] = df["_split"].where(df["_split"].notna(), "train_extra")
    train_mask = df["_split"].isin(["train", "train_extra"]).to_numpy()

    # Reproduce the EXACT QM-scalar standardization train_cross_gnn.py used (fit on train_mask)
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    med = Xdf[train_mask].median(numeric_only=True)
    Xz = Xdf.fillna(med).fillna(0.0)
    qm_mean = Xz[train_mask].mean().to_numpy()
    qm_std = Xz[train_mask].std().replace(0, 1).to_numpy()
    qm_std = np.where(qm_std == 0, 1.0, qm_std)
    qmz = ((Xz.to_numpy() - qm_mean) / qm_std).astype(np.float32)

    test_df = df[df["_split"] == "test"].reset_index(drop=True)
    print(f"test set: {len(test_df)} rows, {test_df['pair_key'].nunique()} pairs")
    test_idx = df.index[df["_split"] == "test"].to_numpy()
    test_qmz = qmz[test_idx]

    # ── GNN predictions ──
    gnn_meta = json.loads((GNN_DIR / "models" / "metadata.json").read_text())
    ad, bd = 40, 6  # verified atom/bond feature dims (see train_cross_gnn.py docstring)
    model = TripleGNN(ad, bd, len(feats), h=128, layers=4)
    sd = torch.load(GNN_DIR / "models" / "cross_gnn_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval()

    te_pairs = build_graphs(test_df.reset_index(drop=True), test_qmz)
    # ym/ysd: the GNN's target standardization must match train_cross_gnn.py's;
    # recompute from the same train-split delta (deterministic, same seed/data)
    train_df = df[df["_split"].isin(["train", "train_extra"])]
    ys = (train_df[TARGET_COL] - train_df[BASELINE_COL]).to_numpy()
    ym, ysd = float(ys.mean()), float(ys.std())
    for p in te_pairs:
        p.ystd = (p.y - ym) / ysd
    mae_gnn, yh_gnn, yt_gnn, ids_gnn = evl(model, make_loader(te_pairs, 64, False), ym, ysd)
    print(f"GNN reproduced frozen holdout MAE={mae_gnn:.3f} (expect ~2.623 from the SLURM log)")

    # ── ensemble predictions ──
    ens = joblib.load(ENSEMBLE_DIR / "models" / "cross_ensemble_model.joblib")
    ens_pred = ens.predict(test_df) + test_df[BASELINE_COL].to_numpy()
    ens_by_id = dict(zip(test_df["id"].astype(str), ens_pred))

    gnn_by_id = dict(zip(ids_gnn, yh_gnn))
    true_by_id = dict(zip(ids_gnn, yt_gnn))

    rows = []
    for _, r in test_df.iterrows():
        i = str(r["id"])
        if i not in gnn_by_id:
            continue
        rows.append({
            "id": i, "pair_key": r["pair_key"], "reaction_type": r["reaction_type"],
            "dG_true": true_by_id[i], "dG_gxtb": r[BASELINE_COL],
            "ens_pred": ens_by_id[i], "gnn_pred": gnn_by_id[i],
        })
    d = pd.DataFrame(rows)
    d["blend_pred"] = 0.5 * d["ens_pred"] + 0.5 * d["gnn_pred"]
    d["err_gxtb"] = (d["dG_gxtb"] - d["dG_true"]).abs()
    d["err_ens"] = (d["ens_pred"] - d["dG_true"]).abs()
    d["err_gnn"] = (d["gnn_pred"] - d["dG_true"]).abs()
    d["err_blend"] = (d["blend_pred"] - d["dG_true"]).abs()
    d = d.sort_values("err_ens", ascending=False).reset_index(drop=True)
    d.to_csv(OUT / "per_row_predictions.csv", index=False)

    print(f"\n=== n={len(d)} point estimates ===")
    print(f"  g-xTB baseline  MAE={d['err_gxtb'].mean():.3f}")
    print(f"  ensemble-only   MAE={d['err_ens'].mean():.3f}")
    print(f"  GNN-only        MAE={d['err_gnn'].mean():.3f}")
    print(f"  blend (w=0.5)   MAE={d['err_blend'].mean():.3f}")

    print(f"\n=== per-row errors, worst-for-ensemble first (n={len(d)}) ===")
    print(d[["id", "reaction_type", "dG_true", "err_gxtb", "err_ens", "err_gnn", "err_blend"]]
          .to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    # ── bootstrap CI on paired deltas (resample ROWS, i.e. n=29 with replacement) ──
    rng = np.random.default_rng(0)
    n = len(d)
    B = 20000
    err_ens = d["err_ens"].to_numpy()
    err_gnn = d["err_gnn"].to_numpy()
    err_blend = d["err_blend"].to_numpy()
    delta_blend_vs_ens = np.zeros(B)
    delta_gnn_vs_ens = np.zeros(B)
    mae_ens_boot = np.zeros(B)
    mae_blend_boot = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        mae_ens_boot[b] = err_ens[idx].mean()
        mae_blend_boot[b] = err_blend[idx].mean()
        delta_blend_vs_ens[b] = err_blend[idx].mean() - err_ens[idx].mean()
        delta_gnn_vs_ens[b] = err_gnn[idx].mean() - err_ens[idx].mean()

    def ci(x, lo=5, hi=95):
        return float(np.percentile(x, lo)), float(np.percentile(x, hi))

    print(f"\n=== bootstrap (B={B}, resampling the same 29 rows with replacement) ===")
    print(f"  ensemble-only MAE:      point={err_ens.mean():.3f}  90% CI={ci(mae_ens_boot)}")
    print(f"  blend MAE:              point={err_blend.mean():.3f}  90% CI={ci(mae_blend_boot)}")
    print(f"  delta (blend - ens):    point={err_blend.mean()-err_ens.mean():+.3f}  "
          f"90% CI={ci(delta_blend_vs_ens)}  "
          f"P(blend better)={float((delta_blend_vs_ens < 0).mean()):.3f}")
    print(f"  delta (gnn-only - ens): point={err_gnn.mean()-err_ens.mean():+.3f}  "
          f"90% CI={ci(delta_gnn_vs_ens)}  "
          f"P(gnn-only better)={float((delta_gnn_vs_ens < 0).mean()):.3f}")

    # ── leave-one-out sensitivity: does removing the single worst row flip the conclusion? ──
    print(f"\n=== leave-worst-row-out sensitivity ===")
    worst_id = d.iloc[0]["id"]
    d_loo = d[d["id"] != worst_id]
    print(f"  dropping row {worst_id} (reaction_type={d.iloc[0]['reaction_type']}, "
          f"dG_true={d.iloc[0]['dG_true']:.1f}, err_ens={d.iloc[0]['err_ens']:.2f}):")
    print(f"    ensemble-only MAE {d['err_ens'].mean():.3f} -> {d_loo['err_ens'].mean():.3f}")
    print(f"    blend MAE       {d['err_blend'].mean():.3f} -> {d_loo['err_blend'].mean():.3f}")
    print(f"    delta (blend-ens) {d['err_blend'].mean()-d['err_ens'].mean():+.3f} -> "
          f"{d_loo['err_blend'].mean()-d_loo['err_ens'].mean():+.3f}")

    summary = {
        "n": int(n),
        "point_mae": {
            "gxtb": float(d["err_gxtb"].mean()), "ensemble_only": float(err_ens.mean()),
            "gnn_only": float(err_gnn.mean()), "blend_w0.5": float(err_blend.mean()),
        },
        "bootstrap_90ci": {
            "ensemble_only_mae": ci(mae_ens_boot), "blend_mae": ci(mae_blend_boot),
            "delta_blend_vs_ens": ci(delta_blend_vs_ens), "delta_gnn_vs_ens": ci(delta_gnn_vs_ens),
        },
        "p_blend_better_than_ensemble": float((delta_blend_vs_ens < 0).mean()),
        "p_gnn_only_better_than_ensemble": float((delta_gnn_vs_ens < 0).mean()),
        "worst_row_loo": {
            "id": worst_id, "delta_before": float(err_blend.mean() - err_ens.mean()),
            "delta_after": float(d_loo["err_blend"].mean() - d_loo["err_ens"].mean()),
        },
    }
    (OUT / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved per-row predictions + bootstrap summary to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
