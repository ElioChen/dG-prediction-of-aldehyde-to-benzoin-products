#!/usr/bin/env python
"""Rec-2 prep: extract archived homo product geometries for the homo_unify_v1 30k.

The 30k homo_unify products' xyz geometries live in
  data/cross_benzoin/bde_homo_product_featurize_20260902/chunk_XXXX/geom.tar.zst
as members  xyz/pNNNNNN.xyz  (chunk-local index). homo_v6/products_all.csv gives
each product id its  chunk_XXXX/xyz/pNNNNNN.xyz  path. This extracts the needed
members into <out>/xyz/<id>.xyz and writes <out>/homo_unify_products_for_mordred.csv
(id, xyz_file, error) in the schema add_mordred_cross_products.py consumes.

  python cross_benzoin/rec2_extract_homo_product_geoms.py --out data/cross_benzoin/homo_unify/product_geoms
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ARCH = REPO / "data/cross_benzoin/bde_homo_product_featurize_20260902"
UNIFY = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_products.csv"
PRODUCTS_ALL = REPO / "data/cross_benzoin/homo_v6/products_all.csv"
_PAT = re.compile(r"(chunk_\d+)/xyz/(p\d+)\.xyz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-chunks", type=int, default=0, help="pilot: only first N chunks")
    args = ap.parse_args()
    xyz_out = args.out / "xyz"
    xyz_out.mkdir(parents=True, exist_ok=True)

    u = pd.read_csv(UNIFY, usecols=["id"])
    u["id"] = u["id"].astype(str)
    pa = pd.read_csv(PRODUCTS_ALL, usecols=["id", "xyz_file", "error"], low_memory=False)
    pa["id"] = pa["id"].astype(str)
    pa = pa[pa["error"].astype("string").fillna("") == ""]
    m = u.merge(pa[["id", "xyz_file"]], on="id", how="left")
    hit = m[m["xyz_file"].notna()].copy()
    miss = m[m["xyz_file"].isna()]["id"].tolist()
    print(f"homo_unify {len(u)}: {len(hit)} with archived xyz path, {len(miss)} missing", flush=True)

    parsed = hit["xyz_file"].str.extract(_PAT)
    hit["chunk"], hit["pmem"] = parsed[0], parsed[1]
    hit = hit[hit["chunk"].notna()]

    chunks = sorted(hit["chunk"].unique())
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]
        hit = hit[hit["chunk"].isin(chunks)]
    print(f"{len(chunks)} chunk archives to read", flush=True)

    rows, n_ok = [], 0
    for i, ck in enumerate(chunks):
        g = hit[hit["chunk"] == ck]
        tar = ARCH / ck / "geom.tar.zst"
        if not tar.exists():
            for _id in g["id"]:
                rows.append({"id": _id, "xyz_file": "", "error": "no_archive"})
            continue
        members = [f"xyz/{p}.xyz" for p in g["pmem"]]
        subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", str(tar),
                        "-C", str(args.out), *members],
                       capture_output=True, text=True)
        for _id, p in zip(g["id"], g["pmem"]):
            src = args.out / "xyz" / f"{p}.xyz"
            dst = xyz_out / f"{_id}.xyz"
            if src.exists():
                if src != dst:
                    src.replace(dst)
                rows.append({"id": _id, "xyz_file": str(dst), "error": ""})
                n_ok += 1
            else:
                rows.append({"id": _id, "xyz_file": "", "error": "extract_fail"})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(chunks)} chunks, {n_ok} xyz extracted", flush=True)

    for _id in miss:
        rows.append({"id": _id, "xyz_file": "", "error": "no_products_all_row"})
    out_csv = args.out / "homo_unify_products_for_mordred.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {n_ok}/{len(u)} products have an extracted geometry", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
