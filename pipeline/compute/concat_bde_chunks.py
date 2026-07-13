#!/usr/bin/env python
"""Concatenate bde_{aldehydes,products} chunk_*.csv into single sidecars.

Only pass --which when the corresponding array is FULLY complete -- writing a partial
sidecar under the final filename would mean overwriting it later when the array finishes,
which conflicts with the never-overwrite-outputs convention. Default (no --which) does both,
for use once both arrays are confirmed done.
"""
import argparse
import pandas as pd, glob
from pathlib import Path

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

ap = argparse.ArgumentParser()
ap.add_argument("--which", choices=["aldehydes", "products"], default=None)
args = ap.parse_args()
whiches = [args.which] if args.which else ["aldehydes", "products"]

for which in whiches:
    files = sorted(glob.glob(str(H / f"bde_{which}" / "chunk_*.csv")))
    if not files:
        print(f"{which}: no chunk files yet, skip", flush=True)
        continue
    expected_rows = sum(len(pd.read_csv(f, usecols=[0])) for f in files)
    out = H / f"{which}_bde_descriptors.csv"
    if out.exists():
        existing_rows = sum(1 for _ in open(out)) - 1
        if existing_rows == expected_rows:
            print(f"{which}: {out} already complete ({existing_rows} rows) -- skip", flush=True)
            continue
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(out, index=False)
    key = "bde_ald_CH_kcal" if which == "aldehydes" else "bde_prod_CC_kcal"
    ok = full[key].notna().sum() if key in full.columns else 0
    print(which, "chunks:", len(files), "rows:", len(full), f"filled({key}):", ok, "-> wrote", out, flush=True)
print("DONE", flush=True)
