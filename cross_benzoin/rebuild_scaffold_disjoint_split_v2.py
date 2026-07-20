#!/usr/bin/env python3
"""
Same method as rebuild_scaffold_disjoint_split.py (single global greedy
bin-packing pass over all scaffolds, largest-first), but with the
molecule-level target changed from 80/10/10 to the user's standing 7:2:1
preference (see memory data-split-721.md), and parameterized so the target
and output path are explicit instead of hardcoded.

Why a new file instead of editing the original: the 80/10/10 target was
inherited from candidates_v3's own original (pre-scaffold-fix) split, not a
choice made during the 2026-07-17 scaffold rebuild -- so this is a genuine
methodology change, not a bugfix, and the original script/output stay as
the historical record (see memory preserve-output-history.md). The pair-level
consequence of this molecule-level ratio change (train shrinks ~24%, val
grows ~3.3x, test roughly unchanged) was explicitly confirmed with the user
before running this for real (2026-07-20).

Usage:
  python cross_benzoin/rebuild_scaffold_disjoint_split_v2.py \
      --train-frac 0.70 --val-frac 0.20 --test-frac 0.10 \
      --out-suffix 721
"""
from __future__ import annotations

import argparse

import pandas as pd

REPO_DATA = "/gpfs/scratch1/shared/schen3/benzoin-dg/data/cross_benzoin/candidates_v3"
IN_PATH = f"{REPO_DATA}/aldehydes_with_scaffold.parquet"
SEED = 20260717


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frac", type=float, required=True)
    ap.add_argument("--val-frac", type=float, required=True)
    ap.add_argument("--test-frac", type=float, required=True)
    ap.add_argument("--out-suffix", required=True, help="appended to output filename, e.g. '721'")
    args = ap.parse_args()

    target_frac = {"train": args.train_frac, "validation": args.val_frac, "test": args.test_frac}
    assert abs(sum(target_frac.values()) - 1.0) < 1e-9, "fractions must sum to 1"
    out_path = f"{REPO_DATA}/aldehydes_with_scaffold_split_{args.out_suffix}.parquet"

    ald = pd.read_parquet(IN_PATH)
    n_scaffolds = ald["scaffold"].nunique()
    print(f"loaded {len(ald)} molecules, {n_scaffolds} distinct scaffolds")
    print(f"target: {target_frac}")

    scaf_counts = ald.groupby("scaffold").size().sort_values(ascending=False)
    n_total = len(ald)
    targets = {k: v * n_total for k, v in target_frac.items()}
    current = {k: 0 for k in target_frac}
    assign = {}

    order = scaf_counts.sample(frac=1.0, random_state=SEED).sort_values(
        ascending=False, kind="stable")
    for scaf, cnt in order.items():
        deficits = {k: targets[k] - current[k] for k in target_frac}
        pick = max(deficits, key=deficits.get)
        assign[scaf] = pick
        current[pick] += cnt

    ald["scaffold_split"] = ald["scaffold"].map(assign)

    print("\n=== overall new scaffold_split sizes ===")
    sizes = ald["scaffold_split"].value_counts()
    print(sizes)
    for k in target_frac:
        print(f"  {k}: {sizes[k]} ({sizes[k]/n_total:.1%}, target {target_frac[k]:.0%})")

    print("\n=== cho_class balance per new split (soft/secondary outcome) ===")
    print(pd.crosstab(ald["scaffold_split"], ald["cho_class"], normalize="index"))

    overlap = ald.groupby("scaffold")["scaffold_split"].nunique()
    assert (overlap == 1).all(), "BUG: some scaffold spans >1 new split"
    print("\nverified: zero scaffold overlap across new scaffold_split")

    ald.to_parquet(out_path)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
