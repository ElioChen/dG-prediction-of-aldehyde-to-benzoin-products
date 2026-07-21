#!/usr/bin/env python3
"""
Re-test homo+cross unification a SECOND way: on the new, honest
scaffold-disjoint split (see memory cross-scaffold-disjoint-rebuild-20260717.md)
instead of the old candidates_v3 molecule-level split / plain CV. Answers: does
adding homo rows help the cross model on a genuinely novel-scaffold test set,
not just on CV (which the round1-7 unification_check_round7.py test already
covers)?

Method: take the SAME 450-row cross-only scaffold-disjoint test set used by
train_scaffold_disjoint.py (never touched, single-XGB reference MAE=2.448).
Build a clean-train pool = cross rows with new_scaffold_split=='train'
(19,687) UNION homo rows whose donor aldehyde's OWN scaffold_split (from
candidates_v3's aldehydes_with_scaffold_split.parquet, homo pairs have
donor==acceptor so a single molecule determines the whole pair) is 'train'
(never 'test'/'validation', to keep the eval set genuinely untouched by any
homo leak). Fit the champion's exact 260-feature single-XGB on the unified
clean-train, evaluate ONCE on the untouched 450-row clean test, compare
against the cross-only reference (2.448).

Usage
  python cross_benzoin/analysis/unification_check_scaffold_disjoint.py \
      --unified-table data/cross_benzoin/homo_unify/cross_train_table_unified_v2_7rounds_mordred.parquet \
      --scaffold-labeled-table data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_split_labeled.parquet \
      --homo-scaffold-lookup data/cross_benzoin/homo_unify/homo_unify_v1_scaffold_split_lookup.csv \
      --feature-list data/cross_benzoin/cross_round7/scaffold_disjoint_v1/models/feature_list.json \
      --cross-only-holdout-mae 2.448 \
      --outdir data/cross_benzoin/cross_round7/unification_check_scaffold_disjoint_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "cross_benzoin"))
import delta_core as dc  # noqa: E402

TARGET_COL = "dG_orca_kcal"
BASELINE_COL = "dG_gxtb_kcal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified-table", type=Path, required=True)
    ap.add_argument("--scaffold-labeled-table", type=Path, required=True)
    ap.add_argument("--homo-scaffold-lookup", type=Path, required=True)
    ap.add_argument("--feature-list", type=Path, required=True)
    ap.add_argument("--cross-only-holdout-mae", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.unified_table)
    is_homo = (df["round"] == "homo_unify_v1").to_numpy()
    print(f"unified table: {len(df)} rows ({(~is_homo).sum()} cross, {is_homo.sum()} homo)")

    # cross rows: pull the official new_scaffold_split label by id
    cross_labels = pd.read_parquet(args.scaffold_labeled_table, columns=["id", "new_scaffold_split"])
    df = df.merge(cross_labels, on="id", how="left", suffixes=("", "_cross"))

    # homo rows: their own aldehyde-level scaffold_split (donor==acceptor for homo)
    homo_lut = pd.read_csv(args.homo_scaffold_lookup, usecols=["id", "scaffold_split"])
    homo_lut["id"] = homo_lut["id"].astype(str)
    df["id"] = df["id"].astype(str)
    df = df.merge(homo_lut, on="id", how="left", suffixes=("", "_homo"))

    # unify into one label column: cross rows use new_scaffold_split, homo rows
    # use their own aldehyde scaffold_split (mutually exclusive by construction:
    # only one of the two merges produced a non-null value per row)
    df["split_label"] = np.where(is_homo, df["scaffold_split"], df["new_scaffold_split"])
    print("split_label counts (cross+homo combined):")
    print(df.groupby(["round" if False else np.where(is_homo, "homo", "cross"), "split_label"]).size())

    clean_train = df[df["split_label"] == "train"].reset_index(drop=True)
    clean_test = df[(~is_homo) & (df["split_label"] == "test")].reset_index(drop=True)
    print(f"\nunified clean-train: {len(clean_train)} rows "
          f"({(clean_train['round'] != 'homo_unify_v1').sum()} cross + "
          f"{(clean_train['round'] == 'homo_unify_v1').sum()} homo)")
    print(f"clean-test (cross-only, untouched, should be 450): {len(clean_test)} rows")

    feats = json.load(open(args.feature_list))
    feats_present = [f for f in feats if f in df.columns]
    missing = [f for f in feats if f not in df.columns]
    print(f"champion feature list: {len(feats)} total, {len(feats_present)} present, {len(missing)} missing")
    if missing:
        print(f"  missing (first 20): {missing[:20]}")

    medians = clean_train[feats_present].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    Xtr = clean_train[feats_present].apply(pd.to_numeric, errors="coerce").fillna(medians)
    Xte = clean_test[feats_present].apply(pd.to_numeric, errors="coerce").fillna(medians)
    ytr = (clean_train[TARGET_COL] - clean_train[BASELINE_COL]).to_numpy()

    m = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, args.seed)
    m.fit(Xtr, ytr)
    pred_test = clean_test[BASELINE_COL].to_numpy() + m.predict(Xte)
    holdout = dc.metrics_vs_dft(clean_test[TARGET_COL].to_numpy(), pred_test)
    base_gxtb = dc.metrics_vs_dft(clean_test[TARGET_COL].to_numpy(), clean_test[BASELINE_COL].to_numpy())

    gain_abs = args.cross_only_holdout_mae - holdout["MAE"]
    gain_rel = gain_abs / args.cross_only_holdout_mae * 100

    result = {
        "n_clean_train_total": len(clean_train),
        "n_clean_train_cross": int((clean_train["round"] != "homo_unify_v1").sum()),
        "n_clean_train_homo": int((clean_train["round"] == "homo_unify_v1").sum()),
        "n_clean_test": len(clean_test),
        "n_features_used": len(feats_present),
        "unified_scaffold_disjoint_holdout_MAE": holdout["MAE"],
        "unified_scaffold_disjoint_holdout_R2": holdout["R2"],
        "gxtb_baseline_holdout_MAE": base_gxtb["MAE"],
        "cross_only_scaffold_disjoint_holdout_MAE_reference": args.cross_only_holdout_mae,
        "unification_gain_abs_kcal": gain_abs,
        "unification_gain_rel_pct": gain_rel,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    (args.outdir / "unification_scaffold_disjoint_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote results -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
