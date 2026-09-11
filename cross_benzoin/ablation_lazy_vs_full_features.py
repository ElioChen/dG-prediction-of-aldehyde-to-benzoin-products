#!/usr/bin/env python
"""Ablation: does the champion need the EXPENSIVE (DFT/xTB-on-product-geometry)
feature groups, or does the LAZY subset (cache/SMILES-derivable, no per-pair
compute) get most of the signal? Directly answers "can the flying dataset be
made genuinely lazy" (CHEMICAL_SPACE.md sec5's correction, 2026-09-11) and
gives a data-backed starting point for a future descriptor redesign.

Same recipe as train_scaffold_disjoint.py's single-XGB champion (n_estimators
300, max_depth 3, lr 0.05, Delta-learning target-baseline, new_scaffold_split
train/test), just varying which columns go into X:

  FULL  = exactly what production trains on: _feature_blocks(...)["all_raw_blocks+mordred"]
  LAZY  = FULL minus (a) PRODUCT_FEATS (product-side QM: mulliken/wbo/fukui/
          vbur/sterimol/hb_*/dih_core/bde -- needs the product's own optimized
          3D geometry) and (b) product_mordred_* (checked: ignore_3D=False,
          several families inherently 3D -- also needs that geometry).
          Donor/acceptor QM+BDE+mordred and all three RDKit-2D blocks stay in
          LAZY: genuinely derivable from the two aldehydes' cached descriptors
          + the generated product SMILES, no compute campaign needed.

Usage
    python cross_benzoin/ablation_lazy_vs_full_features.py \
        --table data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402
from train_cross_delta import _feature_blocks, TARGET_COL, BASELINE_COL  # noqa: E402
from assemble_cross_training_table import PRODUCT_FEATS  # noqa: E402


def run_one(train_df, test_df, feats, seed, label):
    medians = train_df[feats].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    Xtr = train_df[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)
    Xte = test_df[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)
    ytr = (train_df[TARGET_COL] - train_df[BASELINE_COL]).to_numpy()

    m = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, seed)
    m.fit(Xtr, ytr)
    pred_test = test_df[BASELINE_COL].to_numpy() + m.predict(Xte)
    holdout = dc.metrics_vs_dft(test_df[TARGET_COL].to_numpy(), pred_test)
    print(f"[{label}] n_feats={len(feats)}  holdout MAE={holdout['MAE']:.3f}  R2={holdout['R2']:.3f}")
    return holdout, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=5, help="repeat with this many seeds, report mean+-sd")
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    train_df = df[df["new_scaffold_split"] == "train"].reset_index(drop=True)
    test_df = df[df["new_scaffold_split"] == "test"].reset_index(drop=True)
    print(f"loaded {len(df)} rows -> train={len(train_df)} test={len(test_df)} "
          f"(TARGET={TARGET_COL}, BASELINE={BASELINE_COL})")

    all_feats = [c for c in df.columns if c not in {
        "id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
        "donor_smiles", "acceptor_smiles", "smiles", "dG_xtb_kcal", "dG_gxtb_kcal",
        "dG_orca_kcal", "donor_scaf_split", "acceptor_scaf_split", "new_scaffold_split",
        "dG_r2scan_kcal", "dG_b973c_kcal", "dG_gxtb_kcal_selfconsistent",
        "dG_orca_kcal_stored", "label", "label_stored", "gxtb_stored", "grp"}]
    full_feats = _feature_blocks(all_feats)["all_raw_blocks+mordred"]

    not_lazy = set(PRODUCT_FEATS) | {c for c in full_feats if c.startswith("product_mordred_")}
    lazy_feats = [c for c in full_feats if c not in not_lazy]

    print(f"\nFULL feature set: {len(full_feats)} (= production champion's exact feature list)")
    print(f"LAZY feature set: {len(lazy_feats)} "
          f"({len(full_feats) - len(lazy_feats)} dropped: bare PRODUCT_FEATS "
          f"[{sum(1 for c in PRODUCT_FEATS if c in full_feats)}] + product_mordred_* "
          f"[{sum(1 for c in full_feats if c.startswith('product_mordred_'))}])")

    baseline_holdout = dc.metrics_vs_dft(test_df[TARGET_COL].to_numpy(), test_df[BASELINE_COL].to_numpy())
    print(f"\n[baseline, no model] MAE={baseline_holdout['MAE']:.3f}  R2={baseline_holdout['R2']:.3f}")

    results = {"full": [], "lazy": []}
    for seed in range(args.seed, args.seed + args.n_seeds):
        print(f"\n--- seed {seed} ---")
        h_full, _ = run_one(train_df, test_df, full_feats, seed, "FULL (260)")
        h_lazy, _ = run_one(train_df, test_df, lazy_feats, seed, "LAZY (no product QM/mordred)")
        results["full"].append(h_full["MAE"])
        results["lazy"].append(h_lazy["MAE"])

    full_mae = np.array(results["full"])
    lazy_mae = np.array(results["lazy"])
    print(f"\n=== summary over {args.n_seeds} seeds ===")
    print(f"FULL  MAE: {full_mae.mean():.3f} +/- {full_mae.std():.3f}  (seeds: {[f'{v:.3f}' for v in full_mae]})")
    print(f"LAZY  MAE: {lazy_mae.mean():.3f} +/- {lazy_mae.std():.3f}  (seeds: {[f'{v:.3f}' for v in lazy_mae]})")
    delta = lazy_mae.mean() - full_mae.mean()
    pooled_sd = np.sqrt((full_mae.std() ** 2 + lazy_mae.std() ** 2) / 2)
    print(f"delta (lazy - full): {delta:+.3f} kcal/mol  ({delta / full_mae.mean() * 100:+.1f}%), "
          f"seed-sd pooled ~{pooled_sd:.3f} -> {'within noise' if abs(delta) < 2 * pooled_sd else 'likely real'}")
    print(f"g-xTB baseline (no model) MAE: {baseline_holdout['MAE']:.3f} for reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
