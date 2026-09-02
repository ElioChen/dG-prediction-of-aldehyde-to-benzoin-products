#!/usr/bin/env python3
"""
Rebuild `data/cross_benzoin/homo_v6/aldehydes_all.csv`, the aldehyde descriptor
library destroyed by the 2026-07 Snellius purge (`.gitignore` excluded
`/data/cross_benzoin/*/*_all.csv`, so no GitHub branch has it and the old repo's
homo_v6 chunk dirs are empty shells).

Unlike the aldehyde mordred and BDFE blocks -- which were recoverable bit-exactly
from /home/schen3/benzoin_backups -- the xTB/descriptor block had no archive and
had to be recomputed. Two sources are merged here:

  1. `cross_round10_fat20_stage1/chunk_*/aldehydes.csv` -- 15,051 aldehydes emitted
     as a side effect of the round10 featurization (EMIT_ALD=1).
  2. `aldehyde_recovery_r17/chunk_*/aldehydes.csv` -- the aldehydes used by cross
     rounds 1-7 that stage1 did not already cover, recomputed with
     cb_featurize.py --aldehydes-only.

The `id` column is rewritten to the aldehyde library's own key: the **0-based row
index of data/library/aldehydes_clean_v6.csv**. cb_featurize writes an InChIKey
there instead, but every downstream join (ALD_BDE_CSV, ALD_MORDRED_CSV) keys on the
numeric library index and assemble_* coerces it with pd.to_numeric, so an InChIKey
id would silently become NaN and drop the whole aldehyde-side mordred/BDE block.
The index convention was confirmed by reproducing historical `donor_bde_gxtb_kcal`
values exactly. The original InChIKey is preserved as `inchikey`.

Usage:
    python cross_benzoin/rebuild_aldehydes_all.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
SOURCES = [
    REPO / "data/cross_benzoin/cross_round10_fat20_stage1/chunk_*/aldehydes.csv",
    REPO / "data/cross_benzoin/aldehyde_recovery_r17/chunk_*/aldehydes.csv",
]
LIBRARY = REPO / "data/library/aldehydes_clean_v6.csv"
OUT = REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def main() -> int:
    frames = []
    for pat in SOURCES:
        files = sorted(glob.glob(str(pat)))
        if not files:
            print(f"WARN: no files matched {pat}")
            continue
        # a chunk still being written by a running array task has an empty (or
        # header-only) aldehydes.csv -- skip it rather than abort the whole rebuild
        good, skipped = [], 0
        for f in files:
            try:
                good.append(pd.read_csv(f, low_memory=False))
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                skipped += 1
        if not good:
            print(f"WARN: every file under {pat} was empty or unparsable")
            continue
        part = pd.concat(good, ignore_index=True)
        print(f"{Path(pat).parent.parent.name}: {len(files)} chunks "
              f"({skipped} still being written, skipped) -> {len(part)} rows")
        frames.append(part)
    if not frames:
        print("ERROR: no aldehyde sources found", file=sys.stderr)
        return 1

    ald = pd.concat(frames, ignore_index=True)
    err = ald["error"].astype("string").fillna("")
    n_err = (err != "").sum()
    ald = ald[err == ""].copy()
    print(f"combined: {len(ald)} error-free rows ({n_err} dropped with an error)")

    ald = ald.rename(columns={"id": "inchikey"})
    ald["_canon"] = ald["smiles"].map(canon)
    ald = ald[ald["_canon"].notna()].drop_duplicates("_canon")
    print(f"unique canonical aldehydes: {len(ald)}")

    lib = pd.read_csv(LIBRARY, usecols=["SMILES"], low_memory=False)
    lib["id"] = np.arange(len(lib))
    lib["_canon"] = lib["SMILES"].map(canon)
    lib = lib.dropna(subset=["_canon"]).drop_duplicates("_canon")[["_canon", "id"]]
    print(f"library: {len(lib)} unique canonical SMILES")

    ald = ald.merge(lib, on="_canon", how="left")
    n_unmapped = ald["id"].isna().sum()
    print(f"library id mapped: {ald['id'].notna().sum()}/{len(ald)} ({n_unmapped} unmapped)")
    if n_unmapped:
        print("  unmapped aldehydes keep id=NaN; they lose the aldehyde-side mordred/BDE "
              "block but retain every xTB descriptor (the SMILES-keyed lookup still works)")

    cols = ["id", "inchikey", "smiles"] + [c for c in ald.columns
                                            if c not in ("id", "inchikey", "smiles", "_canon")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ald[cols].to_csv(OUT, index=False)
    print(f"wrote {len(ald)} rows x {len(cols)} cols -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
