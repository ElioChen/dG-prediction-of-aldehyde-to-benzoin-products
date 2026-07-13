#!/usr/bin/env python3
"""
Append an expansion batch as a NEW chunk to the canonical training globs, without
touching existing chunks (non-destructive incremental growth).

`delta_core` reads `data/descriptors/chunk_*/descriptors.csv` and
`data/labels/chunk_*/delta_G.csv`, so each labeling batch becomes its own
`chunk_<NN>`. This is the incremental complement to merge_labels.py (which rebuilds
chunk_000): use this for every +N expansion so the join just grows.

Reads (per batch, from the staging area, default = ml/):
  <desc-glob>      e.g. ml/desc_expand/chunk_*/descriptors.csv
  <label-glob>     e.g. ml/labels_expand300/mol_*/delta_G.csv
Writes:
  data/descriptors/chunk_<NN>/descriptors.csv
  data/labels/chunk_<NN>/delta_G.csv     (KEEP cols, rows with a DFT ΔG)

Usage
  python pipeline/add_expand_chunk.py \
      --desc-glob 'ml/desc_expand/chunk_*/descriptors.csv' \
      --label-glob 'ml/labels_expand300/mol_*/delta_G.csv'
  # auto-picks the next free chunk index; override with --chunk N
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LABEL_KEEP = ["index", "aldehyde_smiles", "PubChem_CID",
              "dG_xtb_kcal", "dG_shermo_kcal", "dG_orca_kcal", "dG_orca_shermo_kcal"]


def _concat(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"  {pattern}: {len(files)} files -> {len(df)} rows")
    return df


def _next_chunk(root: Path) -> int:
    used = [int(m.group(1)) for d in root.glob("chunk_*")
            if (m := re.match(r"chunk_(\d+)$", d.name))]
    return (max(used) + 1) if used else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desc-glob", required=True)
    ap.add_argument("--label-glob", required=True)
    ap.add_argument("--chunk", type=int, default=None,
                    help="chunk index; default = next free across both globs")
    ap.add_argument("--desc-root", default=str(REPO / "data/descriptors"))
    ap.add_argument("--label-root", default=str(REPO / "data/labels"))
    args = ap.parse_args()

    desc_root, label_root = Path(args.desc_root), Path(args.label_root)
    chunk = args.chunk
    if chunk is None:
        chunk = max(_next_chunk(desc_root), _next_chunk(label_root))
    tag = f"chunk_{chunk:03d}"

    desc = _concat(args.desc_glob)
    lab = _concat(args.label_glob)

    # Keep only label rows with a DFT ΔG; align to the canonical column set.
    lab["dG_orca_kcal"] = pd.to_numeric(lab.get("dG_orca_kcal"), errors="coerce")
    lab = lab[lab["dG_orca_kcal"].notna()].copy()
    for c in LABEL_KEEP:
        if c not in lab.columns:
            lab[c] = pd.NA
    lab = lab[LABEL_KEEP]

    desc_out = desc_root / tag / "descriptors.csv"
    lab_out = label_root / tag / "delta_G.csv"
    for p in (desc_out, lab_out):
        p.parent.mkdir(parents=True, exist_ok=True)
    desc.columns = [c.strip() for c in desc.columns]   # drop stray EOL whitespace
    desc.to_csv(desc_out, index=False)
    lab.to_csv(lab_out, index=False)

    n_join = desc.merge(lab[["index"]], on="index", how="inner")["index"].nunique()
    print(f"\nWrote {tag}:")
    print(f"  {desc_out}  ({len(desc)} descriptor rows)")
    print(f"  {lab_out}  ({len(lab)} labeled rows, all with DFT ΔG)")
    print(f"  descriptor∩label on index: {n_join} molecules joinable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
