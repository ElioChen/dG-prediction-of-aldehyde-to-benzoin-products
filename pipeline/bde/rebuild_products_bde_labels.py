#!/usr/bin/env python3
"""Rebuild `data/cross_benzoin/homo_v6/products_bdfe_gxtb_descriptors.csv`, the
homo-product g-xTB BDE/BDFE label file destroyed by the 2026-07 Snellius purge
(`.gitignore` excluded `homo_v6/*_descriptors.csv`).

Unlike `products_all.csv` (the local-3D-descriptor library, which has no archive and
must be recomputed), the label file is recoverable bit-for-bit: the per-chunk g-xTB
BDE outputs survived in the home archive
`/gpfs/home4/schen3/benzoin_backups/homo_v6_scratch_archive/bdfe_gxtb_products.tar.gz`
(1,463 chunk CSVs, columns `bdfe_gxtb_kcal, bde_gxtb_kcal, id`). This mirrors how
`aldehydes_bdfe_gxtb_descriptors.csv` was rebuilt during the 2026-09-02 recovery.

`id` is the homo-product key == donor id == the 0-based row index of
`data/library/aldehydes_clean_v6.csv` (same convention as every other homo_v6 file,
see cross_benzoin/rebuild_aldehydes_all.py). Output column order matches the
recovered aldehyde file: `bdfe_gxtb_kcal, id, bde_gxtb_kcal`.

Usage:
    python pipeline/bde/rebuild_products_bde_labels.py
"""
from __future__ import annotations

import sys
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/gpfs/home4/schen3/benzoin_backups/homo_v6_scratch_archive/"
               "bdfe_gxtb_products.tar.gz")
OUT = REPO / "data/cross_benzoin/homo_v6/products_bdfe_gxtb_descriptors.csv"
ALD_REF = REPO / "data/cross_benzoin/homo_v6/aldehydes_bdfe_gxtb_descriptors.csv"


def main() -> int:
    if not ARCHIVE.exists():
        print(f"ERROR: archive not found: {ARCHIVE}", file=sys.stderr)
        return 1

    frames = []
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and m.name.endswith(".csv")]
        for m in members:
            buf = tf.extractfile(m)
            if buf is None:
                continue
            try:
                frames.append(pd.read_csv(BytesIO(buf.read())))
            except pd.errors.EmptyDataError:
                print(f"WARN: empty chunk {m.name}")
    if not frames:
        print("ERROR: no chunks read from archive", file=sys.stderr)
        return 1

    df = pd.concat(frames, ignore_index=True)
    print(f"concatenated {len(frames)} chunks -> {len(df)} rows")

    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    n_before = len(df)
    df = df.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    if len(df) != n_before:
        print(f"dropped {n_before - len(df)} duplicate ids -> {len(df)} unique")

    # match the recovered aldehyde file's column order exactly
    if ALD_REF.exists():
        ald_cols = list(pd.read_csv(ALD_REF, nrows=0).columns)
        cols = [c for c in ald_cols if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
    else:
        cols = ["bdfe_gxtb_kcal", "id", "bde_gxtb_kcal"]
    df = df[cols]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows x {len(cols)} cols -> {OUT.relative_to(REPO)}")
    print(df[["bde_gxtb_kcal", "bdfe_gxtb_kcal"]].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
