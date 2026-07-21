#!/usr/bin/env python
"""Compute the SAME targeted Mordred descriptor families the homo champion uses
(MoRSE/CPSA/Polarizability/GeometricalIndex/MomentOfInertia/PBF/McGowanVolume/
VdwVolumeABC/Weight/TopoPSA -- see pipeline/analysis/finalize_correction_mordred_slim.py's
TARGET_MODULES) for CROSS products, reusing the geometry already saved (xyz_file) by
cb_featurize.py -- no re-optimization needed, matching the reasoning in
cross_benzoin/analysis/add_mordred_descriptors.py (~1s/molecule, real but cheap since no
new simulation is required, only post-processing of an existing optimized geometry).

This is the genuinely-new-compute half of porting homo's descriptor richness to cross
(the aldehyde/donor/acceptor-side 102 slim mordred features are FREE -- see
data/cross_benzoin/homo_v6/aldehydes_mordred_slim102.csv, extracted from the existing
full-220k-library aldehydes_mordred_descriptors.csv with zero new compute).

Usage
  python cross_benzoin/add_mordred_cross_products.py --products-csv <round products.csv> \
      --chunk-id 0 --chunk-size 100 --out-dir <outdir>/mordred_products
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

TARGET_MODULES = {"MoRSE", "CPSA", "Polarizability", "GeometricalIndex", "MomentOfInertia",
                  "PBF", "McGowanVolume", "VdwVolumeABC", "Weight", "TopoPSA"}


def _calc():
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=False)
    keep = [d for d in calc.descriptors if type(d).__module__.split(".")[-1] in TARGET_MODULES]
    return Calculator(keep, ignore_3D=False)


def mol_from_xyz(xyz_file: str):
    if not xyz_file or not Path(xyz_file).exists():
        return None
    mol = Chem.MolFromXYZFile(xyz_file)
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None
    return mol


def one(calc, names, xyz_file: str) -> dict:
    row = {f"mordred_{n}": np.nan for n in names}
    mol = mol_from_xyz(xyz_file)
    if mol is None:
        return row
    try:
        res = calc(mol)
        for n, v in zip(names, res):
            row[f"mordred_{n}"] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else np.nan
    except Exception:
        pass
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products-csv", required=True, type=Path)
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"chunk_{args.chunk_id:04d}.csv"

    df = pd.read_csv(args.products_csv, usecols=["id", "xyz_file", "error"], low_memory=False)
    df = df[df["error"].astype("string").fillna("") == ""].reset_index(drop=True)
    lo, hi = args.chunk_id * args.chunk_size, min((args.chunk_id + 1) * args.chunk_size, len(df))
    if lo >= len(df):
        print(f"chunk {args.chunk_id}: out of range ({lo} >= {len(df)})", flush=True)
        return 0
    sub = df.iloc[lo:hi].reset_index(drop=True)

    if out.exists():
        existing = pd.read_csv(out, usecols=["id"])
        if len(existing) >= len(sub):
            print(f"chunk {args.chunk_id}: already done -- skip", flush=True)
            return 0

    calc = _calc()
    names = [str(d) for d in calc.descriptors]
    print(f"chunk {args.chunk_id}: rows {lo}:{hi} ({len(sub)}), {len(names)} targeted mordred descriptors", flush=True)
    rows = [one(calc, names, xyz) for xyz in sub["xyz_file"]]
    result = pd.concat([sub[["id"]], pd.DataFrame(rows)], axis=1)
    result.to_csv(out, index=False)
    ok = int(result[f"mordred_{names[0]}"].notna().sum())
    print(f"wrote {out} ({len(result)} rows, {ok} with values)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
