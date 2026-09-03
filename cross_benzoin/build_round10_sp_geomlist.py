#!/usr/bin/env python
"""Build an (id, xyz_path) work list for the round10 fat20_stage1 product r2SCAN-3c
SP, from an active-learning selection CSV.

  selection: data/cross_benzoin/cross_round10_fat20_stage1/cross_round10_fat20_stage1_dft_selection.csv
             (score_round_active_learning.py output; `id` = product InChIKey__InChIKey)
  geometry:  data/cross_benzoin/cross_round10_fat20_stage1/cross_round10_products_merged.csv
             (`id`, `xyz_file` absolute path, `error`)

  -> data/cross_benzoin/cross_round10_fat20_stage1/round10_sp_geom_list.csv  (id, xyz_path)

Usage:
  python build_round10_sp_geomlist.py [--limit N]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data/cross_benzoin/cross_round10_fat20_stage1"
SEL = BASE / "cross_round10_fat20_stage1_dft_selection.csv"
PROD = BASE / "cross_round10_products_merged.csv"
OUT = BASE / "round10_sp_geom_list.csv"
OLD_PREFIXES = ("/scratch-shared/schen3/benzoin-dg/", "/gpfs/scratch1/shared/schen3/benzoin-dg/")


def _restore(p: str) -> Path:
    for pre in OLD_PREFIXES:
        if p.startswith(pre):
            return REPO / p[len(pre):]
    return Path(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap the list to the first N rows (0 = no cap)")
    args = ap.parse_args()

    sel = pd.read_csv(SEL)
    prod = pd.read_csv(PROD, low_memory=False)[["id", "xyz_file", "error"]]
    m = sel.merge(prod, on="id", how="left")
    m = m[m["error"].isna() | (m["error"].astype(str).str.strip() == "")]
    m["xyz_path"] = m["xyz_file"].astype(str).map(lambda p: str(_restore(p)))
    m["ok"] = m["xyz_path"].map(lambda p: Path(p).exists())
    have = m[m["ok"]].copy()
    if args.limit and len(have) > args.limit:
        have = have.head(args.limit)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    have[["id", "xyz_path"]].to_csv(OUT, index=False)
    print(f"selection {len(sel)} -> {len(m)} joined -> {len(have)} with geometry on disk "
          f"({'capped to %d' % args.limit if args.limit else 'no cap'})")
    print(f"  -> {OUT}")
    return 0 if len(have) else 1


if __name__ == "__main__":
    raise SystemExit(main())
