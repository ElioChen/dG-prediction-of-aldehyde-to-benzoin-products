#!/usr/bin/env python3
"""
SHAP-slim cross's mordred block, mirroring homo's method
(pipeline/analysis/finalize_correction_mordred_slim.py: rank by descending
mean|SHAP|, greedily keep a feature only if |corr| <= --corr-thresh with every
already-kept feature). Homo did this at 220k rows going 510->199 mordred
feats with no accuracy loss; cross's 423 raw mordred feats were folded in
un-slimmed (see assemble_cross_training_table_v3.py) and risk overfitting at
4-8k row scale -- this closes that specific descriptor-alignment gap.

Reuses the already-trained model from train_cross_delta.py's last run
(cross_delta_model.joblib + feature_list.json) instead of retraining just to
get SHAP values.

Usage
  python cross_benzoin/shap_slim_cross_mordred.py \
      --model-dir data/cross_benzoin/cross_round3/train_3rounds_mordred_v1 \
      --table data/cross_benzoin/cross_round3/cross_train_table_3rounds_mordred.parquet \
      --keep-n 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--keep-n", type=int, default=120,
                     help="target number of mordred feats to keep (upper bound; "
                          "correlation pruning may keep fewer)")
    ap.add_argument("--corr-thresh", type=float, default=0.9)
    ap.add_argument("--out-table", type=Path, default=None)
    args = ap.parse_args()

    model = joblib.load(args.model_dir / "models" / "cross_delta_model.joblib")
    feats = json.loads((args.model_dir / "models" / "feature_list.json").read_text())
    df = pd.read_parquet(args.table)

    mordred_feats = [c for c in feats if "_mordred_" in c]
    base_feats = [c for c in feats if c not in mordred_feats]
    print(f"model has {len(feats)} feats total ({len(base_feats)} base, {len(mordred_feats)} mordred)")

    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xdf.median(numeric_only=True)
    X = Xdf.fillna(medians)

    import shap
    print("computing SHAP values (may take a few minutes at this row count)...")
    expl = shap.Explainer(model, X)
    sv = expl(X)
    mean_abs_shap = np.abs(sv.values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=feats).sort_values(ascending=False)
    importance.to_csv(args.model_dir / "data" / "shap_importance_full.csv", header=["mean_abs_shap"])

    mordred_importance = importance[importance.index.isin(mordred_feats)]
    print(f"\ntop 15 mordred feats by mean|SHAP|:\n{mordred_importance.head(15)}")

    # greedy keep: descending SHAP order, drop any candidate correlated > thresh
    # with an already-kept feature (mirrors homo's finalize_correction_mordred_slim.py)
    Xm = X[mordred_importance.index]
    corr = Xm.corr().abs()
    kept: list[str] = []
    for feat in mordred_importance.index:
        if len(kept) >= args.keep_n:
            break
        if kept and (corr.loc[feat, kept] > args.corr_thresh).any():
            continue
        kept.append(feat)
    print(f"\nkept {len(kept)}/{len(mordred_feats)} mordred feats after SHAP-rank + "
          f"corr>{args.corr_thresh} pruning")

    out_table = args.out_table or (args.table.parent /
                                    f"{args.table.stem}_slim{len(kept)}.parquet")
    meta_cols = ["id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
                 "donor_smiles", "acceptor_smiles", "smiles",
                 "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"]
    drop_mordred = [c for c in mordred_feats if c not in kept]
    out = df.drop(columns=[c for c in drop_mordred if c in df.columns])
    out.to_parquet(out_table, index=False)
    out.to_csv(out_table.with_suffix(".csv"), index=False)
    print(f"wrote slimmed table ({len(out)} rows x {len(out.columns) - len(meta_cols)} feats) -> {out_table}")

    kept_path = args.model_dir / "data" / "mordred_slim_kept.json"
    kept_path.write_text(json.dumps(kept, indent=2))
    print(f"kept feature list -> {kept_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
