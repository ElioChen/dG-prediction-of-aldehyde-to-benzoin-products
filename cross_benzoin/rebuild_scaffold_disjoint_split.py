#!/usr/bin/env python3
"""
Rebuild candidates_v3's molecule split as SCAFFOLD-disjoint instead of just
molecule(InChIKey)-disjoint -- the existing split was found to leak severely
(83.2% of test molecules share a scaffold with something in train; see
memory cross-five-diagnostics-20260717.md).

v2: a per-cho_class-independent scaffold assignment (v1) failed a basic
sanity check -- 2,413 scaffolds (affecting 139,091/220,859 molecules, 63%)
span MORE THAN ONE cho_class, because Bemis-Murcko scaffolds discard the
CHO group and its exact attachment point (the very thing that determines
cho_class), so many different-class molecules share an identical core ring
scaffold. Strict per-category independent assignment is therefore
incompatible with scaffold-disjointness for most of the library.

Fixed: ONE global greedy bin-packing pass over ALL scaffolds (largest-first
by total molecule count), assigning each whole scaffold to whichever split
is furthest below its GLOBAL 80/10/10 target -- guarantees perfect
scaffold-disjointness and close-to-target overall size. cho_class balance
across splits is a secondary, soft outcome (reported, not enforced) rather
than the earlier per-category-independent hard constraint.
"""
from __future__ import annotations

import pandas as pd

REPO_DATA = "/gpfs/scratch1/shared/schen3/benzoin-dg/data/cross_benzoin/candidates_v3"
IN_PATH = f"{REPO_DATA}/aldehydes_with_scaffold.parquet"
OUT_PATH = f"{REPO_DATA}/aldehydes_with_scaffold_split.parquet"

TARGET_FRAC = {"train": 0.80, "validation": 0.10, "test": 0.10}
SEED = 20260717


def main() -> int:
    ald = pd.read_parquet(IN_PATH)
    n_scaffolds = ald["scaffold"].nunique()
    print(f"loaded {len(ald)} molecules, {n_scaffolds} distinct scaffolds")

    multi_class = ald.groupby("scaffold")["cho_class"].nunique()
    n_multi = (multi_class > 1).sum()
    print(f"scaffolds spanning >1 cho_class: {n_multi} ({n_multi/n_scaffolds:.1%}), "
          f"affecting {ald['scaffold'].isin(multi_class[multi_class>1].index).sum()} molecules -- "
          f"confirms per-category-independent assignment is not viable, using single global pass")

    scaf_counts = ald.groupby("scaffold").size().sort_values(ascending=False)
    n_total = len(ald)
    targets = {k: v * n_total for k, v in TARGET_FRAC.items()}
    current = {k: 0 for k in TARGET_FRAC}
    assign = {}

    order = scaf_counts.sample(frac=1.0, random_state=SEED).sort_values(
        ascending=False, kind="stable")
    for scaf, cnt in order.items():
        deficits = {k: targets[k] - current[k] for k in TARGET_FRAC}
        pick = max(deficits, key=deficits.get)
        assign[scaf] = pick
        current[pick] += cnt

    ald["scaffold_split"] = ald["scaffold"].map(assign)

    print("\n=== overall new scaffold_split sizes ===")
    sizes = ald["scaffold_split"].value_counts()
    print(sizes)
    for k in TARGET_FRAC:
        print(f"  {k}: {sizes[k]} ({sizes[k]/n_total:.1%}, target {TARGET_FRAC[k]:.0%})")

    print(f"\n(old, leaky `split` column sizes, for comparison): "
          f"{ald['split'].value_counts().to_dict()}")

    print("\n=== cho_class balance per new split (soft/secondary outcome) ===")
    print(pd.crosstab(ald["scaffold_split"], ald["cho_class"], normalize="index"))

    overlap = ald.groupby("scaffold")["scaffold_split"].nunique()
    assert (overlap == 1).all(), "BUG: some scaffold spans >1 new split"
    print("\nverified: zero scaffold overlap across new scaffold_split")

    ald.to_parquet(OUT_PATH)
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
