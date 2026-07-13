#!/usr/bin/env python3
"""
Assemble per-molecule unified-featurize outputs into ONE training parquet.

The unified `compute/featurize.py` writes one `features.csv` per molecule
(descriptors + dG_xtb + dG_orca on a shared xTB geometry, r2SCAN-3c). This globs
those across batches and writes `data/featurize.parquet` — the single source of
truth `delta_core.load_training_table` trains on.

Replaces the legacy chunk-CSV assembly (legacy/merge_labels.py +
legacy/add_expand_chunk.py). Dedups on `index`, preferring rows that carry a DFT
ΔG. QC (failure filtering) stays in delta_core, so this stays raw.

Usage
  python pipeline/assemble_featurize.py
  python pipeline/assemble_featurize.py --globs 'data/raw/featurize_full/mol_*/features.csv' \
                                                'data/raw/featurize_v3/mol_*/features.csv'
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GLOBS = [
    str(REPO / "data/raw/featurize_full/mol_*/features.csv"),         # original 500
    str(REPO / "data/raw/featurize_v3/mol_*/features.csv"),           # +1000 expansion
    str(REPO / "data/raw/featurize_v4/mol_*/features.csv"),           # +500 clean expansion
    str(REPO / "data/raw/featurize_aromatic_v1/mol_*/features.csv"),  # +813 carbo-aromatic
    str(REPO / "data/raw/featurize_hetero_v1/mol_*/features.csv"),    # +250 hetero-aromatic
]   # NB: featurize_flex (deprecated 7-mol experiment) intentionally excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--globs", nargs="+", default=DEFAULT_GLOBS)
    ap.add_argument("--out", default=str(REPO / "data/featurize.parquet"))
    args = ap.parse_args()

    files: list[str] = []
    for g in args.globs:
        files += sorted(glob.glob(g))
    if not files:
        raise SystemExit(f"No features.csv found under: {args.globs}")

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:                       # skip an in-flight / empty file
            print(f"  skip {f}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df["dG_orca_kcal"] = pd.to_numeric(df.get("dG_orca_kcal"), errors="coerce")

    # Dedup on index; among duplicates keep the one that actually has a DFT ΔG.
    df["_has_orca"] = df["dG_orca_kcal"].notna().astype(int)
    df = (df.sort_values(["index", "_has_orca"])
            .drop_duplicates("index", keep="last")
            .drop(columns="_has_orca")
            .sort_values("index").reset_index(drop=True))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    n_orca = int(df["dG_orca_kcal"].notna().sum())
    print(f"Wrote {args.out}: {len(df)} molecules, {n_orca} with r2SCAN-3c dG_orca "
          f"(from {len(files)} featurize files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
