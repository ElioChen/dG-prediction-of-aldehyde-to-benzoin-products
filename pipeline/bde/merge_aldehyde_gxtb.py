#!/usr/bin/env python3
"""Merge the recompute_aldehyde_gxtb.py array's chunk_*/aldehydes_gxtb.csv outputs
back into data/cross_benzoin/homo_v6/aldehydes_all.csv (fills G_gxtb where it's
currently NaN only -- the 2,718 rows already populated via the old-library backfill
are left untouched, since the recompute's own validation pilot showed it reproduces
those to ~0.1 kcal/mol anyway, no need to touch what's already known-good).

Usage:
  python merge_aldehyde_gxtb.py [--dry-run]
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import pandas as pd
from rdkit import Chem

REPO = Path(__file__).resolve().parents[2]
CHUNK_GLOB = str(REPO / "data/cross_benzoin/bde_homo_product_featurize_20260902/chunk_*/aldehydes_gxtb.csv")
ALD_ALL = REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(CHUNK_GLOB))
    print(f"{len(files)} chunk_*/aldehydes_gxtb.csv files found")
    if not files:
        return 1
    parts = [pd.read_csv(f, low_memory=False) for f in files]
    gxtb = pd.concat(parts, ignore_index=True)
    n_ok = (gxtb["note"] == "ok").sum()
    print(f"{len(gxtb)} rows total, {n_ok} ok, {len(gxtb) - n_ok} failed/no_xyz")
    gxtb = gxtb[gxtb["note"] == "ok"].copy()
    gxtb["_canon"] = gxtb["SMILES"].map(canon)
    gxtb = gxtb.dropna(subset=["_canon"]).drop_duplicates("_canon")
    lookup = gxtb.set_index("_canon")["G_gxtb"]

    ald = pd.read_csv(ALD_ALL, low_memory=False)
    ald["_canon"] = ald["smiles"].map(canon)
    before = ald["G_gxtb"].notna().sum()
    fillable = ald["G_gxtb"].isna() & ald["_canon"].isin(lookup.index)
    ald.loc[fillable, "G_gxtb"] = ald.loc[fillable, "_canon"].map(lookup)
    after = ald["G_gxtb"].notna().sum()
    print(f"G_gxtb coverage: {before}/{len(ald)} -> {after}/{len(ald)} "
          f"(+{after - before}, {fillable.sum()} matched-and-filled)")
    ald = ald.drop(columns=["_canon"])

    if args.dry_run:
        print("[dry-run] not writing")
        return 0
    bak = ALD_ALL.with_suffix(".csv.pre_gxtb_merge_bak")
    if not bak.exists():
        import shutil
        shutil.copy(ALD_ALL, bak)
        print(f"backed up original -> {bak}")
    ald.to_csv(ALD_ALL, index=False)
    print(f"wrote {ALD_ALL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
