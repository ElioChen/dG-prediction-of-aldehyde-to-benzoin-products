#!/usr/bin/env python
"""Flying-dataset build order, step 5 -- "migrate the 35,528 labels"
(CHEMICAL_SPACE.md sec8). Freezes a canonical, ald_idx-addressed table of
every cross pair that currently has a DFT label, so a caller of
`FlyingDataset(known_pairs=...)` (chemical_space.py) no longer has to reach
for an ad-hoc champion training table (which carries 257 feature columns and
a candidates_v3-derived pair_key it doesn't need just to look up a label).

Source of truth: the b973c Tier B table
(cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet)
-- it is a strict superset of the older g-xTB table for this purpose
(same 35,528 pairs, plus dG_r2scan_kcal/dG_b973c_kcal the older table lacks).
Joined to aldehyde_index.parquet by InChIKey (donor_id/acceptor_id), NOT by
canonical-SMILES round-trip (chemical_space.py's own _build_known_lookup uses
SMILES because that's the only key an arbitrary known_pairs table is
guaranteed to carry -- this script controls its own source table, so it can
use the exact key and skip that class of ambiguity entirely): 35,136/35,136
rows resolve on both sides (100%, verified before trusting this join).

Usage
    python cross_benzoin/build_labeled_pairs.py
    -> data/chemical_space/labeled_pairs.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet"
ALDEHYDE_INDEX = REPO / "data/chemical_space/aldehyde_index.parquet"
OUT = REPO / "data/chemical_space/labeled_pairs.parquet"

COLS = ["pair_key", "donor_smiles", "acceptor_smiles", "smiles",
        "new_scaffold_split", "dG_r2scan_kcal", "dG_gxtb_kcal", "dG_b973c_kcal"]


def main() -> int:
    idx = pd.read_parquet(ALDEHYDE_INDEX, columns=["ald_idx", "InChIKey"])
    key_to_ald = dict(zip(idx["InChIKey"], idx["ald_idx"]))

    df = pd.read_parquet(SOURCE, columns=["donor_id", "acceptor_id"] + COLS)
    donor_idx = df["donor_id"].map(key_to_ald)
    acceptor_idx = df["acceptor_id"].map(key_to_ald)
    n_miss = int(donor_idx.isna().sum() + acceptor_idx.isna().sum())
    if n_miss:
        print(f"WARNING: {n_miss} donor/acceptor InChIKeys not found in aldehyde_index "
              f"-- dropping those rows rather than writing a partially-addressed table")
    keep = donor_idx.notna() & acceptor_idx.notna()

    out = pd.DataFrame({
        "donor_ald_idx": donor_idx[keep].astype(int).to_numpy(),
        "acceptor_ald_idx": acceptor_idx[keep].astype(int).to_numpy(),
    })
    for c in COLS:
        out[c] = df.loc[keep, c].to_numpy()
    out = out.rename(columns={"new_scaffold_split": "split", "dG_r2scan_kcal": "label",
                              "smiles": "product_smiles", "dG_gxtb_kcal": "baseline_gxtb_kcal",
                              "dG_b973c_kcal": "baseline_b973c_kcal"})
    out["label_col"] = "dG_r2scan_kcal"
    out["source_table"] = str(SOURCE.relative_to(REPO))

    dup = out.duplicated(["donor_ald_idx", "acceptor_ald_idx"]).sum()
    assert dup == 0, f"{dup} duplicate (donor_ald_idx, acceptor_ald_idx) addresses -- would silently collide on lookup"

    out.to_parquet(OUT, index=False)
    print(f"{len(out)}/{len(df)} pairs addressed and written -> {OUT}")
    print(out["split"].value_counts())
    return 0


if __name__ == "__main__":
    sys.exit(main())
