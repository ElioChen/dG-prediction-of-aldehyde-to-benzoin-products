#!/usr/bin/env python3
"""Assemble the per-chunk output of submit_bde_homo_product_featurize.sh into the two
homo descriptor libraries the 2026-07 purge destroyed:

  data/cross_benzoin/homo_v6/products_all.csv    (fresh full rebuild)
  data/cross_benzoin/homo_v6/aldehydes_all.csv   (fresh full rebuild; the partial 42k
                                                  recovered earlier is only used to
                                                  backfill aldehydes that error here)

`id` convention (confirmed in cross_benzoin/rebuild_aldehydes_all.py): the 0-based row
index of data/library/aldehydes_clean_v6.csv, keyed via canonical SMILES. Neither
featurize_product's features.csv nor its emitted aldehydes.csv carries that id (features.csv
has donor_smiles/product_smiles only; aldehydes.csv's `index` is just a positional
counter), so both are remapped here by canonical SMILES.

Homo product: id == donor_id == parent-aldehyde library index (donor == acceptor).
Product graph SMILES for B6 training = product_smiles -> written as column `smiles`.

Usage:
    python pipeline/bde/assemble_homo_descriptor_libs.py \
        --chunk-dir data/cross_benzoin/bde_homo_product_featurize_20260902 [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]
LIBRARY = REPO / "data/library/aldehydes_clean_v6.csv"
H = REPO / "data/cross_benzoin/homo_v6"
PROD_OUT = H / "products_all.csv"
ALD_OUT = H / "aldehydes_all.csv"
ALD_PARTIAL = H / "aldehydes_all.partial42k.csv"   # renamed copy of the recovered file


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def lib_index() -> pd.DataFrame:
    lib = pd.read_csv(LIBRARY, usecols=["SMILES"], low_memory=False)
    lib["id"] = np.arange(len(lib))
    lib["_canon"] = lib["SMILES"].map(canon)
    return lib.dropna(subset=["_canon"]).drop_duplicates("_canon")[["_canon", "id"]]


def read_chunks(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    good, skipped = [], 0
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if len(df):
                good.append(df)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            skipped += 1
    print(f"  {pattern}\n    {len(files)} files, {skipped} empty/unparsable, "
          f"{sum(len(d) for d in good)} rows")
    return pd.concat(good, ignore_index=True) if good else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cd = args.chunk_dir if args.chunk_dir.is_absolute() else REPO / args.chunk_dir
    if not cd.exists():
        print(f"ERROR: chunk-dir not found: {cd}", file=sys.stderr)
        return 1

    lib = lib_index()
    print(f"library: {len(lib)} unique canonical SMILES\n")

    # ---------------- products ----------------
    print("PRODUCTS")
    prod = read_chunks(str(cd / "chunk_*/features.csv"))
    if prod.empty:
        print("ERROR: no product rows", file=sys.stderr)
        return 1
    err = prod.get("error", pd.Series([""] * len(prod))).astype("string").fillna("")
    n_err = int((err != "").sum())
    prod = prod[err == ""].copy()
    prod = prod.rename(columns={"product_smiles": "smiles"})
    prod["_pcanon"] = prod["smiles"].map(canon)
    prod = prod[prod["_pcanon"].notna()].drop_duplicates("_pcanon")
    prod["_dcanon"] = prod["donor_smiles"].map(canon)
    prod = prod.merge(lib.rename(columns={"_canon": "_dcanon"}), on="_dcanon", how="left")
    prod["donor_id"] = prod["id"]
    n_unmapped = int(prod["id"].isna().sum())
    print(f"  error-free {len(prod)} ({n_err} dropped) | library id mapped "
          f"{int(prod['id'].notna().sum())}/{len(prod)} ({n_unmapped} unmapped)")
    front = [c for c in ["id", "donor_id", "smiles", "donor_smiles", "acceptor_smiles",
                          "error"] if c in prod.columns]
    cols = front + [c for c in prod.columns if c not in front and not c.startswith("_")]
    if args.dry_run:
        print(f"  [dry-run] {len(prod)} x {len(cols)} -> {PROD_OUT}")
    else:
        prod[cols].to_csv(PROD_OUT, index=False)
        print(f"  wrote {len(prod)} x {len(cols)} -> {PROD_OUT.relative_to(REPO)}")

    # ---------------- aldehydes ----------------
    print("\nALDEHYDES")
    ald = read_chunks(str(cd / "chunk_*/aldehydes.csv"))
    if ald.empty:
        print("  no emitted aldehyde rows -> aldehydes_all.csv untouched")
        return 0
    ald = ald.rename(columns={"SMILES": "smiles", "index": "_emit_idx",
                               "G_ald_xtb": "G_xtb"})
    err = ald.get("error", pd.Series([""] * len(ald))).astype("string").fillna("")
    n_err = int((err != "").sum())
    ald = ald[err == ""].copy()
    ald["_canon"] = ald["smiles"].map(canon)
    ald = ald[ald["_canon"].notna()].drop_duplicates("_canon")
    ald = ald.merge(lib, on="_canon", how="left")
    print(f"  error-free unique {len(ald)} ({n_err} dropped) | library id mapped "
          f"{int(ald['id'].notna().sum())}/{len(ald)}")

    # backfill from the partial recovered file for any library id missing here
    partial = ALD_PARTIAL if ALD_PARTIAL.exists() else (
        ALD_OUT if ALD_OUT.exists() else None)
    if partial is not None:
        old = pd.read_csv(partial, low_memory=False)
        have = set(ald["id"].dropna().astype(int))
        add = old[~pd.to_numeric(old["id"], errors="coerce").isin(have)]
        if len(add):
            ald = pd.concat([ald, add], ignore_index=True)
        print(f"  backfilled {len(add)} aldehydes from {partial.name} -> total {len(ald)}")

    front = [c for c in ["id", "inchikey", "smiles"] if c in ald.columns]
    cols = front + [c for c in ald.columns if c not in front and not c.startswith("_")]
    if args.dry_run:
        print(f"  [dry-run] {len(ald)} x {len(cols)} -> {ALD_OUT}")
    else:
        ald[cols].to_csv(ALD_OUT, index=False)
        print(f"  wrote {len(ald)} x {len(cols)} -> {ALD_OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
