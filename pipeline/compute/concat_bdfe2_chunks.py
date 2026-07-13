#!/usr/bin/env python
"""Concatenate bdfe2_{aldehydes,products} chunk_*.csv into single sidecars.
Only pass --which when that side's array is done/near-done -- see concat_bde_chunks.py for
the same never-overwrite-a-partial-file rationale."""
import argparse
import pandas as pd, glob
from pathlib import Path

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

ap = argparse.ArgumentParser()
ap.add_argument("--which", choices=["aldehydes", "products"], default=None)
args = ap.parse_args()
whiches = [args.which] if args.which else ["aldehydes", "products"]

for which in whiches:
    files = sorted(glob.glob(str(H / f"bdfe2_{which}" / "chunk_*.csv")))
    if not files:
        print(f"{which}: no chunk files yet, skip", flush=True)
        continue
    out = H / f"{which}_bdfe2_descriptors.csv"
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(out, index=False)
    ok = full["bdfe_xtb_kcal"].notna().sum() if "bdfe_xtb_kcal" in full.columns else 0
    print(which, "chunks:", len(files), "rows:", len(full), "filled(bdfe_xtb_kcal):", ok, "-> wrote", out, flush=True)
print("DONE", flush=True)
