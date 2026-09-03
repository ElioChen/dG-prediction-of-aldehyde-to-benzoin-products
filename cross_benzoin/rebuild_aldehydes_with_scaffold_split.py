#!/usr/bin/env python
"""Rebuild the purge-lost molecule-level scaffold-split reference

    data/cross_benzoin/candidates_v3/aldehydes_with_scaffold_split.parquet

that relabel_scaffold_split_{8,9}rounds.py read (columns SMILES + scaffold_split;
they Chem.CanonSmiles the SMILES to build a lookup). candidates_v3/ now holds only
133-byte purge stubs. Reconstruct from the two survivors:

  data/cross_benzoin/homo_v6/aldehydes_scaffold_split_from_dG.csv  (id, scaffold, scaffold_split; 80/10/10)
  data/library/aldehydes_clean_v6.csv                              (row index = numeric lib_id -> SMILES, InChIKey)

id in the from_dG file is written "2.0"-style (the known purge float-id artefact);
coerce via int(float(x)). Run with an env that has pyarrow (/home/schen3/venv/nequip).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "data/cross_benzoin/homo_v6/aldehydes_scaffold_split_from_dG.csv"
CLEAN = REPO / "data/library/aldehydes_clean_v6.csv"
OUT = REPO / "data/cross_benzoin/candidates_v3/aldehydes_with_scaffold_split.parquet"


def main() -> int:
    sp = pd.read_csv(SPLIT)
    sp["id"] = sp["id"].map(lambda x: int(float(x)))
    clean = pd.read_csv(CLEAN, usecols=["SMILES", "InChIKey"])
    clean["id"] = range(len(clean))

    m = sp.merge(clean, on="id", how="inner")
    miss = len(sp) - len(m)
    m = m[m["SMILES"].notna() & (m["SMILES"].astype(str).str.len() > 0)].copy()

    out = m[["id", "SMILES", "InChIKey", "scaffold", "scaffold_split"]].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    vc = out["scaffold_split"].value_counts().to_dict()
    print(f"{len(sp)} split rows, {len(clean)} clean_v6 rows -> {len(out)} joined "
          f"({miss} unmatched id, dropped blank-SMILES too)")
    print(f"  scaffold_split: {vc}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
