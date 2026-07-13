#!/usr/bin/env python
"""Flag broken-topology benzoin geometries in a DFT results dir.

The DFT runs use the legacy unguarded conformer search, which relaxes ~8.5% of benzoin
products into broken-connectivity geometries (bond formed/broken/fragmented) whose
xTB/DFT energies are artifacts. This scans the chunk CSVs + benzoin_xyz/ of a results
dir, compares each product's perceived heavy-atom topology to its SMILES graph, and
writes a per-molecule flag CSV so downstream fits can exclude/flag them.

Usage:
  python flag_broken_topology.py --results <dir> --out <flags.csv> [--workers 24]
"""
import sys, os, argparse, glob, types
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compute"))
# break thermo_orca circular import (we only need conf_crest's topo helpers)
sys.modules.setdefault("conf_funnel_v3", types.ModuleType("conf_funnel_v3"))
from conf_crest import _ref_topo, _xyz_topo
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")


def _one(args):
    smiles, xyz_path = args
    if not smiles or not os.path.exists(xyz_path):
        return None
    try:
        ref = _ref_topo(smiles)
        got = _xyz_topo(open(xyz_path).read())
        return None if got is None else (got != ref)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="dir with chunk_*.csv and benzoin_xyz/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    chunks = sorted(glob.glob(f"{args.results}/chunk_*.csv"))
    if not chunks:
        print("no chunk CSVs in", args.results); return 1
    df = pd.concat([pd.read_csv(c) for c in chunks], ignore_index=True)
    need = {"benzoin_smiles", "benzoin_xyz_file"}
    if not need.issubset(df.columns):
        print("missing columns; have:", list(df.columns)); return 1

    xyzdir = f"{args.results}/benzoin_xyz"
    tasks = [(r.get("benzoin_smiles"),
              f"{xyzdir}/{os.path.basename(str(r.get('benzoin_xyz_file','')))}")
             for _, r in df.iterrows()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        flags = list(ex.map(_one, tasks, chunksize=64))
    df["broken_topology"] = flags
    ok = df.dropna(subset=["broken_topology"]).copy()
    ok["broken_topology"] = ok["broken_topology"].astype(bool)

    keep = [c for c in ["idx", "index", "aldehyde_smiles", "benzoin_smiles",
                        "dG_xtb_kcal", "dG_orca_kcal", "broken_topology"] if c in df.columns]
    df[keep].to_csv(args.out, index=False)
    n = len(ok)
    print(f"rows: {len(df)}  checked: {n}  "
          f"broken: {int(ok['broken_topology'].sum())} ({100*ok['broken_topology'].mean():.1f}%)")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
