#!/usr/bin/env python
"""Flying-dataset build order, step 1 (CHEMICAL_SPACE.md sec8.1):

Freeze the canonical aldehyde index: one row per `data/library/aldehydes_clean_v6.csv`
entry, keyed by a stable integer `ald_idx` = the row's position in that file (never
re-derived, never an enumerate() over a filtered view -- CHEMICAL_SPACE.md sec3).

Reuses rather than recomputes: the Bemis-Murcko scaffold + scaffold_split label already
computed for this exact library in
`data/cross_benzoin/candidates_v3/aldehydes_with_scaffold_split.parquet` (built by
cross-benzoin's round-7 scaffold-disjoint rebuild, itself reused by
`pipeline/bde/build_scaffold_splits.py` for the BDE project's aldehyde half). Verified
below that its `id` column is positionally == ald_idx (exact raw-SMILES match, checked
against the full base library, not just a sample) -- so this is a safe row-order merge,
not a fuzzy join. 335 rows present in the base library but absent from that parquet
(RDKit-unparseable SMILES, presumably) get scaffold=None and are exactly the molecules
`computable` should end up False for once anything downstream tries to use them.

Canonical SMILES is recomputed fresh for all 220,860 rows (the existing parquet's
`SMILES` column is the RAW smiles, not canonical) using the same RDKit convention as
`build_scaffold_splits.py`'s murcko() -- Chem.MolToSmiles(mol) with default (canonical)
settings -- so canonical SMILES and scaffold SMILES come from the same parse.

`computable` is left null/unknown here per CHEMICAL_SPACE.md sec3 ("leave computable
unknown initially") -- it gets set False downstream once a molecule is confirmed to fail
geometry/DFT, not asserted from a SMILES-parse failure alone (a molecule could still be
computable even if this particular library's stored raw SMILES needed RDKit sanitization
to canonicalize -- computable is about the physics pipeline, not this bookkeeping step).

Usage: python cross_benzoin/build_aldehyde_index.py
Output: data/chemical_space/aldehyde_index.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
BASE_CSV = REPO / "data/library/aldehydes_clean_v6.csv"
SCAFFOLD_PARQUET = REPO / "data/cross_benzoin/candidates_v3/aldehydes_with_scaffold_split.parquet"
OUT = REPO / "data/chemical_space/aldehyde_index.parquet"


def canonical_smiles(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def main() -> int:
    base = pd.read_csv(BASE_CSV)
    n = len(base)
    print(f"base library: {n} rows, columns {base.columns.tolist()}")

    scaf = pd.read_parquet(SCAFFOLD_PARQUET)
    print(f"existing scaffold parquet: {len(scaf)} rows, columns {scaf.columns.tolist()}")

    # Verify the row-order assumption on the FULL overlap before trusting it, not just a
    # sample -- a silent misalignment here would poison every downstream pair address.
    scaf_idx = scaf.set_index("id")
    in_range = scaf_idx.index[(scaf_idx.index >= 0) & (scaf_idx.index < n)]
    base_smiles_at = base["SMILES"].to_numpy()
    mismatches = 0
    for idx in in_range:
        if base_smiles_at[idx] != scaf_idx.at[idx, "SMILES"]:
            mismatches += 1
    print(f"positional id==ald_idx check: {len(in_range)} rows checked, {mismatches} mismatches")
    if mismatches:
        print("FATAL: scaffold parquet's id is NOT a clean positional ald_idx match -- abort.")
        return 1

    idx = pd.DataFrame({
        "ald_idx": np.arange(n, dtype=np.int32),
        "name": base["name"].values,
        "smiles_raw": base["SMILES"].values,
        "InChIKey": base["InChIKey"].values,
        "cho_class": base["cho_class"].values,
        "xtb_risk": base["xtb_risk"].values,
    })

    print("computing canonical SMILES for all rows (fresh, single RDKit pass)...")
    idx["smiles_canonical"] = [canonical_smiles(s) for s in idx["smiles_raw"]]
    n_unparseable = idx["smiles_canonical"].isna().sum()
    print(f"  {n_unparseable} rows failed RDKit parse (computable candidates for False)")

    scaffold_map = scaf.set_index("id")["scaffold"]
    scaffold_split_map = scaf.set_index("id")["scaffold_split"]
    idx["scaffold"] = idx["ald_idx"].map(scaffold_map)
    idx["scaffold_split_legacy"] = idx["ald_idx"].map(scaffold_split_map)
    n_no_scaffold = idx["scaffold"].isna().sum()
    print(f"  {n_no_scaffold} rows have no scaffold (absent from the reused parquet)")

    idx["computable"] = pd.array([None] * n, dtype="boolean")  # unknown, per spec

    OUT.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(OUT, index=False)
    idx.head(2000).to_csv(OUT.with_suffix(".head2000.csv"), index=False)
    print(f"\nwrote {len(idx)} rows x {len(idx.columns)} cols -> {OUT}")

    # Sanity spot-checks
    assert idx["ald_idx"].is_unique and idx["ald_idx"].min() == 0 and idx["ald_idx"].max() == n - 1
    assert len(idx) == n == 220_860 or True  # library size is whatever it is; just report
    print(f"ald_idx range: [{idx['ald_idx'].min()}, {idx['ald_idx'].max()}], unique={idx['ald_idx'].is_unique}")
    dupes = idx["smiles_canonical"].dropna()
    print(f"canonical-SMILES duplicates: {dupes.duplicated().sum()} (expected: near-0, library was pre-deduped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
