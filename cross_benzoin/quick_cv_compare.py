#!/usr/bin/env python3
"""Quick, standalone repeated-grouped-CV comparison: does adding a named block of
extra features to the current 150-feature cross table improve CV MAE? Bypasses
train_cross_delta.py's _feature_blocks() (which only recognizes its own hardcoded
column-name patterns) so new column families (e.g. mordred_*) can be tested
immediately without touching the production training script.

Usage
  python cross_benzoin/quick_cv_compare.py --table <augmented.parquet> \
      --base-table <original.parquet> --extra-prefix donor_mordred_ acceptor_mordred_ \
      --folds 5 --repeats 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cross_delta import repeated_group_cv  # noqa: E402

META = {"id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
        "donor_smiles", "acceptor_smiles", "smiles",
        "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-table", type=Path, required=True)
    ap.add_argument("--augmented-table", type=Path, required=True)
    ap.add_argument("--extra-prefix", nargs="+", required=True,
                     help="column name prefixes that mark the NEW features to test")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = pd.read_parquet(args.base_table)
    aug = pd.read_parquet(args.augmented_table)
    print(f"base: {len(base)} rows; augmented: {len(aug)} rows")

    base_feats = [c for c in base.columns if c not in META]
    extra_feats = [c for c in aug.columns if c not in META and c not in base_feats
                   and any(c.startswith(p) for p in args.extra_prefix)]
    print(f"base feats: {len(base_feats)}; new extra feats: {len(extra_feats)}")

    groups = aug["pair_key"].to_numpy()

    print("\n=== baseline (current 150 feats) ===")
    delta, base_gxtb, _, _, _ = repeated_group_cv(aug, base_feats, groups, args.folds, args.repeats, args.seed)
    print(f"  MAE={delta['MAE']:.3f} RMSE={delta['RMSE']:.3f} R2={delta['R2']:.3f}  (g-xTB MAE={base_gxtb['MAE']:.3f})")

    print(f"\n=== baseline + {len(extra_feats)} new feats ===")
    delta2, base_gxtb2, _, _, _ = repeated_group_cv(aug, base_feats + extra_feats, groups, args.folds, args.repeats, args.seed)
    print(f"  MAE={delta2['MAE']:.3f} RMSE={delta2['RMSE']:.3f} R2={delta2['R2']:.3f}  (g-xTB MAE={base_gxtb2['MAE']:.3f})")

    print(f"\nDelta: {delta['MAE'] - delta2['MAE']:+.3f} kcal/mol MAE change (positive = improvement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
