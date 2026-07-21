#!/usr/bin/env python
"""Product-side g-xTB BDE (ketC-carbC bond) for CROSS-benzoin products.

This is the descriptor missing from the cross Δ-model: donor-/acceptor-side BDE was
free (reused aldehydes_bdfe_gxtb_descriptors.csv, already computed for the full 220k
library), but no cross PRODUCT has ever had its own new-C-C-bond BDE computed -- that
compute is inherently product-specific (a different molecule per donor/acceptor pair).

Recipe (matches the promoted homo feature's method, cheap tier -- BDE only, no BDFE/
Hessian, per [[bde-descriptor-idea]] "SHAP: BDE >> BDFE, drop BDFE"):
  1. Locate ketC/carbC via featurize_product.find_benzoin_core (purely geometric --
     works identically for homo or cross products, same function cb_featurize.py
     used at feature-compute time, so the bond identification is consistent).
  2. Split at that bond (calc_bde.py's split_at_bond), GFN2 --opt tight (uhf=1) each
     radical fragment (calc_bde.py's _run_xtb_opt) -- no Hessian needed for BDE.
  3. g-xTB single point (COSMO/DMSO) on each optimized fragment -- same call cb_featurize
     uses for the parent (_gxtb_sp), extended with --uhf 1 for the open-shell fragments.
  4. Parent E_gxtb is NOT recomputed: cb_featurize.py already stored G_gxtb (free energy)
     plus G_xtb/xtb_energy (GFN2 free/electronic) for every product, so the parent's own
     electronic g-xTB energy is recovered for free:
         E_gxtb_parent = G_gxtb - (G_xtb - xtb_energy)
     (same G_gxtb = E_gxtb + (G_gfn2 - E_el_gfn2) identity cb_featurize.py itself uses).
  BDE_gxtb_kcal = (E_gxtb_fragA + E_gxtb_fragB - E_gxtb_parent) * HARTREE_TO_KCAL

Usage
  python calc_bde_gxtb_product_cross.py --products-csv .../cross_pilot_v1_products.csv \
      --out .../cross_pilot_v1_bde_gxtb.csv [--n 20] [--chunk-id 0 --chunk-size 100 --out-dir ...]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ald_descriptors_qm as A                                    # noqa: E402
import featurize_product as FP                                    # noqa: E402
from calc_bde import _run_xtb_opt, _xyz_block, mol_with_bonds, split_at_bond  # noqa: E402

HARTREE_TO_KCAL = 627.509474
GXTB_BIN_DEFAULT = "/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb"


def _gxtb_sp_uhf(xyz_str: str, work_dir: Path, gxtb_bin: str, charge: int, uhf: int,
                 solvent=("cosmo", "dmso"), timeout: int = 900) -> float | None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "g.xyz").write_text(xyz_str, encoding="utf-8")
    cmd = [gxtb_bin, "g.xyz", "--gxtb", "--sp", "--chrg", str(charge), "--uhf", str(uhf)]
    if solvent:
        cmd += ["--" + solvent[0], *solvent[1:]]
    try:
        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout + r.stderr
    import re
    m = re.findall(r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh", out)
    return float(m[-1]) if m else None


def bde_gxtb_product_cc(xyz_file: str, e_gxtb_parent: float, xtb_bin: str, gxtb_bin: str,
                        work_dir: Path) -> dict:
    row = {"bde_gxtb_kcal": np.nan, "E_gxtb_parent": e_gxtb_parent,
           "E_gxtb_fragA": np.nan, "E_gxtb_fragB": np.nan, "note": ""}
    if e_gxtb_parent is None or not np.isfinite(e_gxtb_parent):
        row["note"] = "missing_parent_e_gxtb"
        return row
    xyz = Path(xyz_file).read_text()
    symbols, coords = A.parse_xyz(xyz)
    core = FP.find_benzoin_core(symbols, coords)
    if core is None:
        row["note"] = "benzoin_core_not_found"
        return row
    i, j = core["ketC"], core["carbC"]
    mol = mol_with_bonds(xyz_file)
    if mol is None:
        row["note"] = "determine_bonds_failed"
        return row
    split = split_at_bond(mol, i, j, coords, symbols)
    if split is None:
        row["note"] = "split_failed_ring_bond"
        return row
    (symA, coordA), (symB, coordB) = split

    xyzA, xyzB = _xyz_block(symA, coordA), _xyz_block(symB, coordB)
    optA = _run_xtb_opt(xyzA, work_dir / "fragA_gfn2", xtb_bin, charge=0, uhf=1)
    optB = _run_xtb_opt(xyzB, work_dir / "fragB_gfn2", xtb_bin, charge=0, uhf=1)
    optA_xyz = work_dir / "fragA_gfn2" / "xtbopt.xyz"
    optB_xyz = work_dir / "fragB_gfn2" / "xtbopt.xyz"
    if optA is None or optB is None or not optA_xyz.exists() or not optB_xyz.exists():
        row["note"] = "gfn2_fragment_opt_failed"
        return row

    e_gxtb_a = _gxtb_sp_uhf(optA_xyz.read_text(), work_dir / "fragA_gxtb", gxtb_bin, charge=0, uhf=1)
    e_gxtb_b = _gxtb_sp_uhf(optB_xyz.read_text(), work_dir / "fragB_gxtb", gxtb_bin, charge=0, uhf=1)
    row["E_gxtb_fragA"], row["E_gxtb_fragB"] = e_gxtb_a, e_gxtb_b
    if e_gxtb_a is None or e_gxtb_b is None:
        row["note"] = "gxtb_fragment_sp_failed"
        return row
    row["bde_gxtb_kcal"] = round((e_gxtb_a + e_gxtb_b - e_gxtb_parent) * HARTREE_TO_KCAL, 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products-csv", required=True)
    ap.add_argument("--n", type=int, default=None, help="pilot mode: first N rows")
    ap.add_argument("--chunk-id", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--out", default=None, help="pilot mode: single output CSV")
    ap.add_argument("--out-dir", default=None, help="array mode: chunk_NNNN.csv written here")
    ap.add_argument("--xtb-bin", default="/home/schen3/xtb/bin/xtb")
    ap.add_argument("--gxtb-bin", default=GXTB_BIN_DEFAULT)
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.products_csv, low_memory=False)
    df = df[df["error"].astype("string").fillna("") == ""].reset_index(drop=True)
    df["E_gxtb_parent"] = df["G_gxtb"] - (df["G_xtb"] - df["xtb_energy"])

    if args.chunk_id is not None:
        lo, hi = args.chunk_id * args.chunk_size, min((args.chunk_id + 1) * args.chunk_size, len(df))
        if lo >= len(df):
            print(f"chunk {args.chunk_id}: out of range"); return 0
        chunk = df.iloc[lo:hi].reset_index(drop=True)
        out_path = Path(args.out_dir) / f"chunk_{args.chunk_id:04d}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and len(pd.read_csv(out_path)) >= len(chunk):
            print(f"chunk {args.chunk_id}: already done -- skip"); return 0
        print(f"chunk {args.chunk_id}: rows {lo}:{hi} ({len(chunk)})")
    else:
        chunk = df.head(args.n or 20).reset_index(drop=True)
        out_path = Path(args.out)
        print(f"piloting product BDE_gxtb on {len(chunk)} rows")

    wd_root = Path(args.work_dir or "/tmp/bde_gxtb_cross_pilot")
    rows = []
    for _, rec in chunk.iterrows():
        wd = wd_root / f"m{rec['id']}"
        try:
            r = bde_gxtb_product_cc(rec["xyz_file"], rec["E_gxtb_parent"],
                                    args.xtb_bin, args.gxtb_bin, wd)
        except Exception as e:
            r = {"bde_gxtb_kcal": np.nan, "note": f"exception:{e}"}
        r["id"] = rec["id"]
        rows.append(r)
        print(rec["id"], r, flush=True)
        shutil.rmtree(wd, ignore_errors=True)

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    ok = result["bde_gxtb_kcal"].notna().sum()
    print(f"wrote {out_path} ({ok}/{len(result)} succeeded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
