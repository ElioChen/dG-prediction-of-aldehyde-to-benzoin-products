#!/usr/bin/env python3
"""
Decisive version of learning_curve_check.py: holds the CURRENT production
architecture fixed (mordred features via the "all_raw_blocks+mordred" block,
MLP+XGB ensemble model via repeated_group_cv_ensemble) while varying the
training-data fraction, instead of the old script's fixed single-XGB
"all+interaction" block. Answers: under the architecture that actually
produced round1-6's real CV MAE gains, is more data still helping, or has
*that* plateaued too?

Usage
  python cross_benzoin/learning_curve_check_ensemble.py \
      --table data/cross_benzoin/cross_round6/cross_train_table_6rounds_mordred_slim120_matched.parquet
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
from train_cross_delta import _feature_blocks  # noqa: E402
from train_cross_ensemble import repeated_group_cv_ensemble  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    all_pairs = df["pair_key"].unique()
    rng = np.random.default_rng(args.seed)
    shuffled_pairs = rng.permutation(all_pairs)

    all_feats = [c for c in df.columns if c not in {
        "id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
        "donor_smiles", "acceptor_smiles", "smiles",
        "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"}]
    feats = _feature_blocks(all_feats)["all_raw_blocks+mordred"]

    print(f"{len(all_pairs)} unique pairs, {len(df)} rows, {len(feats)} features "
          f"(all_raw_blocks+mordred, MLP+XGB ensemble)\n")
    print(f"{'frac':>6}{'n_pairs':>10}{'n_rows':>10}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    rows = []
    for frac in args.fractions:
        n_pairs = max(10, int(round(frac * len(all_pairs))))
        sub_pairs = set(shuffled_pairs[:n_pairs])
        sub = df[df["pair_key"].isin(sub_pairs)].reset_index(drop=True)
        groups = sub["pair_key"].to_numpy()
        delta, base_gxtb, _, _, _ = repeated_group_cv_ensemble(
            sub, feats, groups, args.folds, args.repeats, args.seed)
        print(f"{frac:>6.2f}{n_pairs:>10}{len(sub):>10}{delta['MAE']:>8.3f}"
              f"{delta['RMSE']:>8.3f}{delta['R2']:>8.3f}   "
              f"(g-xTB MAE={base_gxtb['MAE']:.3f})")
        rows.append({"frac": frac, "n_pairs": n_pairs, "n_rows": len(sub),
                     "MAE": delta["MAE"], "RMSE": delta["RMSE"], "R2": delta["R2"],
                     "gxtb_MAE": base_gxtb["MAE"]})

    out_csv = args.table.parent / "learning_curve_ensemble.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
