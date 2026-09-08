#!/usr/bin/env python
"""Restore the full homo product Mordred library from the 2026-07-15 home backup.

`~/benzoin_backups/homo_v6_scratch_archive/mordred_products.tar.gz` holds the
pre-purge full-library product Mordred (2196 chunk CSVs, mordred_* columns,
1826 descriptors) -- so the missing homo_v6/products_mordred_descriptors.csv
needs NO recompute, only decompress + concat + dedup by id.

Writes:
  homo_v6/products_mordred_full.parquet   -- every column, deduped by id
  homo_v6/products_mordred_descriptors.csv -- same rows, id + mordred_* (assembler input)

  python cross_benzoin/rec2_restore_homo_product_mordred.py --workdir <scratch>
"""
from __future__ import annotations
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path.home() / "benzoin_backups/homo_v6_scratch_archive/mordred_products.tar.gz"
OUT_DIR = REPO / "data/cross_benzoin/homo_v6"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    ex = args.workdir / "mordred_products"
    args.workdir.mkdir(parents=True, exist_ok=True)

    if not list(ex.glob("chunk_*.csv")):
        print(f"extracting {ARCHIVE} -> {args.workdir}", flush=True)
        subprocess.run(["tar", "xzf", str(ARCHIVE), "-C", str(args.workdir)], check=True)
    files = sorted(ex.glob("chunk_*.csv"))
    print(f"{len(files)} chunk CSVs", flush=True)

    frames = []
    for i, f in enumerate(files):
        frames.append(pd.read_csv(f, low_memory=False))
        if (i + 1) % 400 == 0:
            print(f"  read {i + 1}/{len(files)}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    print(f"concat {len(df)} rows, {df.shape[1]} cols; unique id {df['id'].nunique()}", flush=True)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).drop_duplicates("id")
    df["id"] = df["id"].astype("int64")
    val = int(df.filter(like="mordred_").notna().any(axis=1).sum())
    print(f"after dedup: {len(df)} rows, {val} with any mordred value", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq = OUT_DIR / "products_mordred_full.parquet"
    csv = OUT_DIR / "products_mordred_descriptors.csv"
    df.to_parquet(pq, index=False)
    keep = ["id"] + [c for c in df.columns if c.startswith("mordred_")]
    df[keep].to_csv(csv, index=False)
    print(f"wrote {pq} ({len(df)}x{df.shape[1]})", flush=True)
    print(f"wrote {csv} ({len(df)}x{len(keep)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
