#!/usr/bin/env python3
"""
Merge round8's per-chunk products.csv (written by cb_featurize.py via the
submit_cb_featurize_array.sh array, job 24707568) into one
cross_round8_products.csv, matching the pattern used by
cross_benzoin/slurm/pilot_gate_autofire.sh's gate 4/5 python block and every
prior round's manual merge step.

Usage:
    python cross_benzoin/merge_round8_products.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RDIR = REPO / "data/cross_benzoin/cross_round8"


def main() -> int:
    fs = sorted(glob.glob(str(RDIR / "chunk_*/products.csv")))
    if not fs:
        print("ERROR: no chunk_*/products.csv found", file=sys.stderr)
        return 1
    dfs = [pd.read_csv(f, low_memory=False) for f in fs]
    df = pd.concat(dfs, ignore_index=True)
    n_err = (df["error"].astype("string").fillna("") != "").sum() if "error" in df.columns else 0
    out = RDIR / "cross_round8_products.csv"
    df.to_csv(out, index=False)
    print(f"merged {len(fs)} chunk files -> {len(df)} rows ({n_err} with error) -> {out}")

    # also merge aldehydes.csv emitted per-chunk (--emit-aldehydes), for completeness /
    # future inspection -- assemble_cross_round_features.py itself reuses the homo_v6
    # cache directly and does not need this file, but keep it for parity with other rounds.
    afs = sorted(glob.glob(str(RDIR / "chunk_*/aldehydes.csv")))
    if afs:
        adf = pd.concat([pd.read_csv(f, low_memory=False) for f in afs], ignore_index=True)
        aout = RDIR / "cross_round8_aldehydes.csv"
        adf.to_csv(aout, index=False)
        print(f"merged {len(afs)} chunk aldehyde files -> {len(adf)} rows -> {aout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
