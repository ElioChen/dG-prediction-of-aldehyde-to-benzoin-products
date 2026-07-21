#!/usr/bin/env python3
"""
Re-test homo+cross unification at round1-7 scale (32,456 cross rows), mirroring
assemble_cross_training_table_unified_v2.py's round1-5 methodology (see memory
cross-ensemble-shipped-and-unification-shrinking-20260716.md): fit ONE mixed
CV (cross round1-7 + the same unchanged 30k-row stratified homo_unify_v1
sample) on the champion's exact 260-feature list, then restrict the resulting
OOF predictions to cross-only rows and compare that MAE against the cross-only
model's own standalone CV MAE (round1-7 champion: 2.122, from
train_7rounds_mordred_slim120_v1/models/metadata.json).

Usage
  python cross_benzoin/analysis/unification_check_round7.py \
      --unified-table data/cross_benzoin/homo_unify/cross_train_table_unified_v2_7rounds_mordred.parquet \
      --feature-list data/cross_benzoin/cross_round7/train_7rounds_mordred_slim120_v1/models/feature_list.json \
      --cross-only-cv-mae 2.122136494840926 \
      --outdir data/cross_benzoin/cross_round7/unification_check_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "cross_benzoin"))
import delta_core as dc  # noqa: E402

TARGET_COL = "dG_orca_kcal"
BASELINE_COL = "dG_gxtb_kcal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified-table", type=Path, required=True)
    ap.add_argument("--feature-list", type=Path, required=True)
    ap.add_argument("--cross-only-cv-mae", type=float, required=True,
                     help="reference: the cross-only champion's own standalone CV MAE")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.unified_table)
    print(f"unified table: {len(df)} rows, round counts:\n{df['round'].value_counts().to_string()}")

    feats = json.load(open(args.feature_list))
    feats_present = [f for f in feats if f in df.columns]
    missing = [f for f in feats if f not in df.columns]
    print(f"champion feature list: {len(feats)} total, {len(feats_present)} present in unified table, "
          f"{len(missing)} missing")
    if missing:
        print(f"  missing (first 20): {missing[:20]}")

    is_homo = df["round"] == "homo_unify_v1"
    print(f"cross rows: {(~is_homo).sum()}, homo rows: {is_homo.sum()}")

    Xdf = df[feats_present].apply(pd.to_numeric, errors="coerce")
    medians = Xdf.median(numeric_only=True)
    X = Xdf.fillna(medians)
    y = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()
    groups = df["pair_key"].to_numpy()

    oof_sum = np.zeros(len(df))
    oof_cnt = np.zeros(len(df))
    for r in range(args.repeats):
        gkf = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + r)
        for tr, te in gkf.split(X, y, groups=groups):
            m = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3,
                                       "learning_rate": 0.05}, args.seed)
            m.fit(X.iloc[tr], y[tr])
            oof_sum[te] += m.predict(X.iloc[te])
            oof_cnt[te] += 1
    oof = oof_sum / np.maximum(oof_cnt, 1)
    pred_dft = df[BASELINE_COL].to_numpy() + oof
    abs_err = np.abs(pred_dft - df[TARGET_COL].to_numpy())

    out = df[["id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round"]].copy()
    out["dG_orca_kcal"] = df[TARGET_COL]
    out["dG_gxtb_kcal"] = df[BASELINE_COL]
    out["dG_pred"] = pred_dft
    out["abs_err"] = abs_err
    out.to_csv(args.outdir / "unified_cv_predictions.csv", index=False)

    mae_all = float(abs_err.mean())
    mae_cross_only = float(abs_err[~is_homo.to_numpy()].mean())
    mae_homo_only = float(abs_err[is_homo.to_numpy()].mean())
    n_cross = int((~is_homo).sum())
    n_homo = int(is_homo.sum())

    gain_abs = args.cross_only_cv_mae - mae_cross_only
    gain_rel = gain_abs / args.cross_only_cv_mae * 100

    result = {
        "n_unified_total": len(df),
        "n_cross_rows": n_cross,
        "n_homo_rows": n_homo,
        "n_features_used": len(feats_present),
        "n_features_missing_from_table": len(missing),
        "unified_cv_mae_all_rows": mae_all,
        "unified_cv_mae_cross_rows_only": mae_cross_only,
        "unified_cv_mae_homo_rows_only": mae_homo_only,
        "cross_only_standalone_cv_mae_reference": args.cross_only_cv_mae,
        "unification_gain_abs_kcal": gain_abs,
        "unification_gain_rel_pct": gain_rel,
        "verdict": "helpful" if gain_abs > 0 else "harmful",
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    (args.outdir / "unification_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote results -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
