#!/usr/bin/env python
"""Concatenate mordred_{products,aldehydes} chunk_*.csv into single wide sidecars."""
import pandas as pd, glob
from pathlib import Path

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

for which in ["products", "aldehydes"]:
    files = sorted(glob.glob(str(H / f"mordred_{which}" / "chunk_*.csv")))
    expected_rows = sum(len(pd.read_csv(f, usecols=[0])) for f in files)
    out = H / f"{which}_mordred_descriptors.csv"
    # resume-safe: a previous run (e.g. killed by a login-node task timeout) may have left
    # a truncated file -- only trust it if the row count matches the source chunks exactly.
    if out.exists():
        existing_rows = sum(1 for _ in open(out)) - 1
        if existing_rows == expected_rows:
            print(f"{which}: {out} already complete ({existing_rows} rows) -- skip", flush=True)
            continue
        print(f"{which}: {out} exists but incomplete ({existing_rows}/{expected_rows}) -- redo", flush=True)
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(out, index=False)
    print(which, "chunks:", len(files), "rows:", len(full), "cols:", full.shape[1], "-> wrote", out, flush=True)
print("DONE", flush=True)
