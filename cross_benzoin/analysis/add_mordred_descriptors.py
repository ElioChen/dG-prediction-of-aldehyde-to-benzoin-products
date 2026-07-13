#!/usr/bin/env python
"""Compute the full Mordred descriptor set (1826, incl. 3D: WHIM/GETAWAY/MoRSE/RDF/volume)
for a CHUNK of a homo_v6 table, reusing the geometry already saved (xyz_file) -- no
re-optimization. Bonds are perceived from the 3D geometry via rdDetermineBonds (xyz has no
bond info). Chunked (not one big job): Mordred is ~0.97s/molecule (vs RDKit's much cheaper
2D-only calc), so the full 220k x2 library is split into array tasks for wall-clock speed.

Usage:
  python add_mordred_descriptors.py --which products --chunk-id 0 --chunk-size 250 \
      --out-dir data/cross_benzoin/homo_v6/mordred_products
  python add_mordred_descriptors.py --which aldehydes --chunk-id 0 --chunk-size 250 \
      --out-dir data/cross_benzoin/homo_v6/mordred_aldehydes

Writes <out-dir>/chunk_<NNNN>.csv (id + mordred_<name> columns). Resume-safe: skips a
chunk whose output file already exists with the right row count.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from mordred import Calculator, descriptors

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

R = Path("/scratch-shared/schen3/benzoin-dg")
H = R / "data/cross_benzoin/homo_v6"
CALC = Calculator(descriptors, ignore_3D=False)
NAMES = [str(d) for d in CALC.descriptors]


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


def one(rec) -> dict:
    row = {f"mordred_{n}": np.nan for n in NAMES}
    mol = mol_from_xyz(rec.get("xyz_file"))
    if mol is None:
        return row
    try:
        res = CALC(mol)
        for n, v in zip(NAMES, res):
            row[f"mordred_{n}"] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else np.nan
    except Exception:
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["products", "aldehydes"], required=True)
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src = H / f"{args.which}_all.csv"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"chunk_{args.chunk_id:04d}.csv"

    df = pd.read_csv(src, usecols=["id", "xyz_file"], dtype=str, keep_default_na=False, low_memory=False)
    df = df.drop_duplicates("id").reset_index(drop=True)
    lo, hi = args.chunk_id * args.chunk_size, min((args.chunk_id + 1) * args.chunk_size, len(df))
    if lo >= len(df):
        print(f"chunk {args.chunk_id}: out of range ({lo} >= {len(df)}), nothing to do", flush=True)
        return
    sub = df.iloc[lo:hi].reset_index(drop=True)

    if out.exists():
        existing = pd.read_csv(out, usecols=["id"])
        if len(existing) >= len(sub):
            print(f"chunk {args.chunk_id}: already done ({len(existing)}/{len(sub)}) — skip", flush=True)
            return

    # NB: intra-task multiprocessing (ProcessPoolExecutor) was tried and DEADLOCKS here --
    # near-zero CPU usage while "running" for minutes. Classic fork()+threaded-BLAS hang
    # (numpy/scipy's OpenBLAS thread pool doesn't survive fork once initialized in the
    # parent). Fix: serial within a task, rely on SLURM array-level parallelism (%128)
    # for wall-clock speed instead -- each array task is a fresh process, no fork issue.
    print(f"{args.which} chunk {args.chunk_id}: rows {lo}:{hi} ({len(sub)})", flush=True)
    rows = [one(rec) for rec in sub.to_dict("records")]
    result = pd.concat([sub[["id"]], pd.DataFrame(rows)], axis=1)
    result.to_csv(out, index=False)
    ok = int(result[f"mordred_{NAMES[0]}"].notna().sum())
    print(f"wrote {out}  ({len(result)} rows, {ok} with mordred values)", flush=True)


if __name__ == "__main__":
    main()
