#!/usr/bin/env python3
"""
Diagnostic (no new DFT compute): is the cross-benzoin Δ-model's CV MAE still
improving with more data, or has it plateaued? Subsamples the 3-round
combined training table at increasing fractions (by UNIQUE PAIR, keeping
both orientations of any sampled pair together -- consistent with every CV
split elsewhere in this project), refits the "all+interaction" feature block
via the same repeated_group_cv used for the real metric (fewer repeats here,
this is a trend check not a final number), and reports the MAE/RMSE/R2 trend.

Answers: is round 4's ~1800-row DFT-SP campaign likely worth the cost, or are
we in a flat part of the curve where more data of this same kind won't move
the needle much?

Usage
  python cross_benzoin/learning_curve_check.py \
      --table data/cross_benzoin/cross_round3/cross_train_table_3rounds.parquet
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
from train_cross_delta import _feature_blocks, repeated_group_cv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
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
    feats = _feature_blocks(all_feats)["all+interaction"]

    print(f"{len(all_pairs)} unique pairs, {len(df)} rows, {len(feats)} features "
          f"(all+interaction)\n")
    print(f"{'frac':>6}{'n_pairs':>10}{'n_rows':>10}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    rows = []
    for frac in args.fractions:
        n_pairs = max(10, int(round(frac * len(all_pairs))))
        sub_pairs = set(shuffled_pairs[:n_pairs])
        sub = df[df["pair_key"].isin(sub_pairs)].reset_index(drop=True)
        groups = sub["pair_key"].to_numpy()
        delta, base_gxtb, _, _, _ = repeated_group_cv(
            sub, feats, groups, args.folds, args.repeats, args.seed)
        print(f"{frac:>6.2f}{n_pairs:>10}{len(sub):>10}{delta['MAE']:>8.3f}"
              f"{delta['RMSE']:>8.3f}{delta['R2']:>8.3f}   "
              f"(g-xTB MAE={base_gxtb['MAE']:.3f})")
        rows.append({"frac": frac, "n_pairs": n_pairs, "n_rows": len(sub),
                     "MAE": delta["MAE"], "RMSE": delta["RMSE"], "R2": delta["R2"],
                     "gxtb_MAE": base_gxtb["MAE"]})

    out_csv = args.table.parent / "learning_curve.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
