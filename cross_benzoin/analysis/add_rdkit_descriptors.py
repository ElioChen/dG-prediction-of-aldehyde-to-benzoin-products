#!/usr/bin/env python
"""Compute the full RDKit 2D descriptor set for a homo_v6 table and write it as a
non-destructive sidecar keyed by `id` (join back with pandas.merge on 'id').

The main products/aldehydes CSVs hold only xTB/geometry/Multiwfn (QM) descriptors — no
RDKit-level 2D descriptors. The correction models compute 16 hand-picked RDKit globals
on the fly but never persist them. This adds ALL RDKit 2D descriptors (217 in RDKit
2026.03.1) once, so future work can just merge them in.

Usage:
  python add_rdkit_descriptors.py products   # -> products_rdkit_descriptors.csv
  python add_rdkit_descriptors.py aldehydes  # -> aldehydes_rdkit_descriptors.csv
"""
import os
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path("data/cross_benzoin/homo_v6")
NAMES = [n for n, _ in Descriptors.descList]           # 217 descriptors
CALC = MoleculeDescriptors.MolecularDescriptorCalculator(NAMES)


def _row(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return [np.nan] * len(NAMES)
    try:
        vals = CALC.CalcDescriptors(m)
        return [v if np.isfinite(v) else np.nan for v in vals]
    except Exception:
        return [np.nan] * len(NAMES)


def main(which):
    if which == "products":
        src, smi_col, out = ROOT / "products_all.csv", "smiles", ROOT / "products_rdkit_descriptors.csv"
    elif which == "aldehydes":
        src, smi_col, out = ROOT / "aldehydes_all.csv", "smiles", ROOT / "aldehydes_rdkit_descriptors.csv"
    else:
        sys.exit("arg must be 'products' or 'aldehydes'")

    df = pd.read_csv(src, usecols=["id", smi_col], dtype=str,
                     keep_default_na=False, low_memory=False)
    df = df[df[smi_col].str.strip() != ""].drop_duplicates("id").reset_index(drop=True)
    print(f"{which}: {len(df):,} rows with SMILES, computing {len(NAMES)} RDKit descriptors", flush=True)

    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("NPROC") or 16)
    with Pool(nproc) as pool:
        mat = pool.map(_row, df[smi_col].tolist(), chunksize=256)

    desc = pd.DataFrame(mat, columns=[f"rdkit_{n}" for n in NAMES])
    result = pd.concat([df[["id", smi_col]].reset_index(drop=True), desc], axis=1)
    result.to_csv(out, index=False)
    ok = int(desc.iloc[:, 0].notna().sum())
    print(f"wrote {out}  ({len(result):,} rows x {len(NAMES)} RDKit cols; "
          f"{ok:,} parsed, {len(result)-ok:,} unparsable SMILES)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "products")
