#!/usr/bin/env python3
"""
Merge ΔG label sources into one clean training table for the pipeline.

Combines, in priority order (later overrides earlier on duplicate `index`, keeping
the row that actually has a DFT ΔG):
  * the reconstructed main run        (data/labels/delta_G_main.csv)
  * the parallel-ORCA supplement      (ml/labels_supp/delta_G.csv)
  * per-molecule array tasks          (<expand>/mol_*/delta_G.csv)

Writes data/labels/chunk_000/delta_G.csv so it matches the pipeline's default
label glob (data/labels/chunk_*/delta_G.csv).

Usage
  python pipeline/merge_labels.py
  python pipeline/merge_labels.py --array-glob 'ml/labels_expand/mol_*/delta_G.csv'
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
KEEP = ["index", "aldehyde_smiles", "PubChem_CID",
        "dG_xtb_kcal", "dG_shermo_kcal", "dG_orca_kcal"]


def _load(path_or_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(path_or_glob))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  skip {f}: {e}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        print(f"  {path_or_glob}: {len(files)} file(s) -> {len(df)} rows")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default=str(REPO / "data/labels/delta_G_main.csv"))
    ap.add_argument("--array-glob", default=str(REPO / "data/raw/labels/mol_*/delta_G.csv"))
    ap.add_argument("--out", default=str(REPO / "data/labels/chunk_000/delta_G.csv"))
    args = ap.parse_args()

    print("Loading label sources...")
    parts = [_load(args.main), _load(args.array_glob)]
    allrows = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if allrows.empty:
        print("No label rows found."); return 1

    for c in KEEP:
        if c not in allrows.columns:
            allrows[c] = pd.NA
    allrows["index"] = pd.to_numeric(allrows["index"], errors="coerce")
    allrows["dG_orca_kcal"] = pd.to_numeric(allrows["dG_orca_kcal"], errors="coerce")

    # Prefer rows that have a DFT ΔG; among those, last wins (supplement/array override).
    allrows["_has_orca"] = allrows["dG_orca_kcal"].notna().astype(int)
    allrows = (allrows.sort_values(["index", "_has_orca"])
               .drop_duplicates("index", keep="last"))
    final = allrows[KEEP].sort_values("index").reset_index(drop=True)

    n_orca = final["dG_orca_kcal"].notna().sum()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out, index=False)
    print(f"\nMerged: {len(final)} molecules, {n_orca} with DFT ΔG "
          f"({len(final)-n_orca} still missing)")
    print(f"Wrote {args.out}")
    if len(final) - n_orca:
        miss = final[final["dG_orca_kcal"].isna()]["index"].tolist()
        print(f"Missing ORCA ΔG for: {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
