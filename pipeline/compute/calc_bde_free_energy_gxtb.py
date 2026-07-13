#!/usr/bin/env python
"""BDFE, g-xTB-CONSISTENT variant: the GFN2-level BDFE (calc_bde_free_energy_v2.py) gave a
null result for the main correction model (see bde-descriptor-idea memory). But the main
model's baseline is g-xTB (dG_gxtb), not GFN2 -- a method mismatch between the BDE/BDFE
descriptor (GFN2) and the quantity it's meant to help correct (g-xTB errors) may be diluting
the signal. This script computes a method-consistent g-xTB BDFE using this project's own
established hybrid-correction pattern (gxtb_baseline.py): G_gxtb = E_gxtb + (G_gfn2 - E_gfn2),
i.e. g-xTB electronic energy + GFN2's RRHO thermal correction (g-xTB itself has no native
production thermal-correction pipeline in this project; only geometry+SP were validated,
see gxtb-solv-pilot-campaign memory -- though --gxtb --ohess --cosmo works fine per a smoke
test, using the SAME hybrid pattern as the rest of this project keeps this new descriptor
consistent with dG_gxtb itself, which is what actually matters for it to be diagnostic of
g-xTB's own errors).

Parent's G_gxtb is ALREADY CACHED (G_gxtb column in {aldehydes,products}_all.csv, computed
by gxtb_baseline.py) -- no new parent calc needed. Only the two radical FRAGMENTS need:
  1. GFN2 --ohess (as in v2, for the RRHO thermal correction AND to relax the fragment
     geometry -- xtbopt.xyz)
  2. g-xTB --sp --cosmo dmso on that SAME relaxed geometry (cheap, no new Hessian)
  G_gxtb_frag = E_gxtb_frag + (G_gfn2_frag - E_gfn2_frag)
  BDFE_gxtb = G_gxtb_fragA + G_gxtb_fragB - G_gxtb_parent(cached)

This redoes the (expensive) fragment ohess step from scratch -- v2's intermediate fragment
E/G values were never saved, only the final combined bde_E_kcal/bdfe_xtb_kcal (per-molecule
scratch dirs are deleted immediately after each task, per this project's scratch-hygiene
discipline) -- so there's no way to reuse those results even though this is "the same"
fragment computation. PILOT FIRST before considering a full-library array (same cost as the
original v2 run, ~1 day of core-hours, plus a modest g-xTB SP addendum).

PILOT usage:
  python calc_bde_free_energy_gxtb.py --which aldehydes --n 20 --out /tmp/bdfe_gxtb_pilot_ald.csv
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ald_descriptors_qm as A
import featurize_product as FP
import thermo_orca as Th
from calc_bde_free_energy_v2 import g_h_atom_correction, H_ATOM_G_CORR, mol_with_bonds, split_at_bond, _xyz_block

RDLogger.DisableLog("rdApp.*")
HARTREE_TO_KCAL = 627.509474


def _run_ohess_uhf(xyz_str, work_dir, xtb_bin, charge, uhf, solvent="dmso", T=298.15, P_atm=1.0, timeout=1800):
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "mol.xyz").write_text(xyz_str, encoding="utf-8")
    Th._write_xtb_inp(work_dir / "xtb.inp", T, P_atm * 1.01325)
    cmd = [xtb_bin, "mol.xyz", "--gfn", "2", "--ohess", "tight", "--input", "xtb.inp",
          "--chrg", str(charge), "--uhf", str(uhf), "--norestart", "--parallel", "1"]
    if solvent:
        cmd += ["--alpb", solvent]
    stdout = ""
    try:
        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout)
        stdout = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        pass
    return stdout


def _gxtb_sp(xyz_file, work_dir, gxtb_bin, charge, uhf, solvent="dmso", timeout=300):
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(xyz_file, work_dir / "g.xyz")
    cmd = [gxtb_bin, "g.xyz", "--gxtb", "--sp", "--chrg", str(charge), "--uhf", str(uhf)]
    if solvent:
        cmd += ["--cosmo", solvent]
    try:
        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout)
        return A._parse_xtb_energy(r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return None


def _frag_G_E_gxtb(symbols, coords, work_dir, xtb_bin, gxtb_bin, solvent="dmso") -> dict:
    """Fresh dual-level calc for a RADICAL fragment: GFN2 --ohess (thermal correction +
    relaxed geometry) followed by a g-xTB SP on that SAME relaxed geometry."""
    if len(symbols) == 1:  # isolated H atom -- no relaxation needed, no ohess possible
        wd = work_dir / "h_sp"; wd.mkdir(parents=True, exist_ok=True)
        (wd / "h.xyz").write_text(_xyz_block(symbols, coords))
        e_gxtb = _gxtb_sp(wd / "h.xyz", wd / "gxtb", gxtb_bin, charge=0, uhf=1, solvent=solvent)
        if e_gxtb is None:
            return {"G_gxtb": None, "E_gxtb": None}
        return {"G_gxtb": e_gxtb + H_ATOM_G_CORR, "E_gxtb": e_gxtb}  # method-independent ideal-gas correction

    ohess_dir = work_dir / "ohess"
    stdout = _run_ohess_uhf(_xyz_block(symbols, coords), ohess_dir, xtb_bin, charge=0, uhf=1, solvent=solvent)
    e_gfn2 = Th._parse_xtb_energy(stdout)
    g_gfn2 = Th.parse_xtb_G(stdout)
    opt_xyz = ohess_dir / "xtbopt.xyz"
    if e_gfn2 is None or g_gfn2 is None or not opt_xyz.exists():
        return {"G_gxtb": None, "E_gxtb": None}
    e_gxtb = _gxtb_sp(opt_xyz, work_dir / "gxtb", gxtb_bin, charge=0, uhf=1, solvent=solvent)
    if e_gxtb is None:
        return {"G_gxtb": None, "E_gxtb": None}
    g_gxtb = e_gxtb + (g_gfn2 - e_gfn2)  # project's established hybrid-correction pattern
    return {"G_gxtb": g_gxtb, "E_gxtb": e_gxtb}


def bdfe_gxtb_aldehyde_ch(xyz_file, parent_G_gxtb, parent_E_gxtb, xtb_bin, gxtb_bin, work_dir) -> dict:
    row = {"bdfe_gxtb_kcal": np.nan, "bde_gxtb_kcal": np.nan}
    if parent_G_gxtb is None:
        return row
    xyz = Path(xyz_file).read_text()
    symbols, coords = A.parse_xyz(xyz)
    hits = A.find_aldehyde_atoms(symbols, coords)
    if not hits:
        return row
    c_idx, o_idx, _ = hits[0]
    dist = np.linalg.norm(coords - coords[c_idx], axis=1)
    h_idx = next((k for k in range(len(symbols))
                 if symbols[k] == "H" and dist[k] < A.CH_BOND_MAX and k != c_idx), None)
    if h_idx is None:
        return row
    mol = mol_with_bonds(xyz_file)
    if mol is None:
        return row
    split = split_at_bond(mol, c_idx, h_idx, coords, symbols)
    if split is None:
        return row
    (symA, coordA), (symB, coordB) = split
    if len(symB) != 1:
        (symA, coordA), (symB, coordB) = (symB, coordB), (symA, coordA)
    if len(symB) != 1:
        return row

    fragA = _frag_G_E_gxtb(symA, coordA, work_dir / "fragA", xtb_bin, gxtb_bin)
    fragB = _frag_G_E_gxtb(symB, coordB, work_dir / "fragB", xtb_bin, gxtb_bin)
    if None not in (fragA["G_gxtb"], fragB["G_gxtb"]):
        row["bdfe_gxtb_kcal"] = round((fragA["G_gxtb"] + fragB["G_gxtb"] - parent_G_gxtb) * HARTREE_TO_KCAL, 3)
    if parent_E_gxtb is not None and None not in (fragA["E_gxtb"], fragB["E_gxtb"]):
        row["bde_gxtb_kcal"] = round((fragA["E_gxtb"] + fragB["E_gxtb"] - parent_E_gxtb) * HARTREE_TO_KCAL, 3)
    return row


def bdfe_gxtb_product_cc(xyz_file, parent_G_gxtb, parent_E_gxtb, xtb_bin, gxtb_bin, work_dir) -> dict:
    row = {"bdfe_gxtb_kcal": np.nan, "bde_gxtb_kcal": np.nan}
    if parent_G_gxtb is None:
        return row
    xyz = Path(xyz_file).read_text()
    symbols, coords = A.parse_xyz(xyz)
    core = FP.find_benzoin_core(symbols, coords)
    if core is None:
        return row
    i, j = core["ketC"], core["carbC"]
    mol = mol_with_bonds(xyz_file)
    if mol is None:
        return row
    split = split_at_bond(mol, i, j, coords, symbols)
    if split is None:
        return row
    (symA, coordA), (symB, coordB) = split

    fragA = _frag_G_E_gxtb(symA, coordA, work_dir / "fragA", xtb_bin, gxtb_bin)
    fragB = _frag_G_E_gxtb(symB, coordB, work_dir / "fragB", xtb_bin, gxtb_bin)
    if None not in (fragA["G_gxtb"], fragB["G_gxtb"]):
        row["bdfe_gxtb_kcal"] = round((fragA["G_gxtb"] + fragB["G_gxtb"] - parent_G_gxtb) * HARTREE_TO_KCAL, 3)
    if parent_E_gxtb is not None and None not in (fragA["E_gxtb"], fragB["E_gxtb"]):
        row["bde_gxtb_kcal"] = round((fragA["E_gxtb"] + fragB["E_gxtb"] - parent_E_gxtb) * HARTREE_TO_KCAL, 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--chunk-id", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=150)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--xtb-bin", default="/home/schen3/xtb/bin/xtb")
    ap.add_argument("--gxtb-bin", default="/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
    src = H / f"{args.which}_all.csv"
    # G_xtb/xtb_energy (GFN2 G/E, cached from the original funnel_v3 featurization) let us
    # algebraically recover the parent's raw g-xTB electronic energy for FREE, no new xtb
    # call needed: E_gxtb = G_gxtb - (G_gfn2 - E_gfn2), inverting gxtb_baseline.py's own
    # G_gxtb = E_gxtb + (G_gfn2 - E_gfn2) hybrid-correction formula. This gives raw-E
    # bde_gxtb_kcal alongside bdfe_gxtb_kcal at essentially no extra compute cost.
    df = pd.read_csv(src, usecols=["id", "xyz_file", "error", "G_gxtb", "G_xtb", "xtb_energy"], dtype=str,
                     keep_default_na=False, low_memory=False)
    df = df[(df["error"] == "") & (df["xyz_file"] != "") & (df["G_gxtb"] != "")
           & (df["G_xtb"] != "") & (df["xtb_energy"] != "")]
    df = df.drop_duplicates("id").reset_index(drop=True)
    df["G_gxtb"] = df["G_gxtb"].astype(float)
    df["G_xtb"] = df["G_xtb"].astype(float)
    df["xtb_energy"] = df["xtb_energy"].astype(float)
    df["E_gxtb_parent"] = df["G_gxtb"] - (df["G_xtb"] - df["xtb_energy"])

    if args.chunk_id is not None:
        lo, hi = args.chunk_id * args.chunk_size, min((args.chunk_id + 1) * args.chunk_size, len(df))
        if lo >= len(df):
            print(f"chunk {args.chunk_id}: out of range, nothing to do", flush=True); return
        df = df.iloc[lo:hi].reset_index(drop=True)
        out_path = Path(args.out_dir) / f"chunk_{args.chunk_id:04d}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            existing = pd.read_csv(out_path, usecols=["id"])
            if len(existing) >= len(df):
                print(f"chunk {args.chunk_id}: already done ({len(existing)}/{len(df)}) -- skip", flush=True)
                return
        print(f"BDFE-gxtb {args.which} chunk {args.chunk_id}: rows {lo}:{hi} ({len(df)})", flush=True)
    else:
        df = df.head(args.n or 15)
        out_path = Path(args.out)
        print(f"BDFE-gxtb pilot on {len(df)} {args.which}", flush=True)

    wd_root = Path(args.work_dir or "/tmp/bdfe_gxtb_pilot")
    rows = []
    for _, rec in df.iterrows():
        wd = wd_root / f"m{rec['id']}"
        try:
            if args.which == "aldehydes":
                r = bdfe_gxtb_aldehyde_ch(rec["xyz_file"], rec["G_gxtb"], rec["E_gxtb_parent"], args.xtb_bin, args.gxtb_bin, wd)
            else:
                r = bdfe_gxtb_product_cc(rec["xyz_file"], rec["G_gxtb"], rec["E_gxtb_parent"], args.xtb_bin, args.gxtb_bin, wd)
        except Exception as e:
            r = {"error": str(e)}
        r["id"] = rec["id"]
        rows.append(r)
        print(rec["id"], r, flush=True)
        shutil.rmtree(wd, ignore_errors=True)

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    ok_g = result["bdfe_gxtb_kcal"].notna().sum() if "bdfe_gxtb_kcal" in result else 0
    ok_e = result["bde_gxtb_kcal"].notna().sum() if "bde_gxtb_kcal" in result else 0
    print(f"wrote {out_path}  bdfe_gxtb:{ok_g}/{len(result)}  bde_gxtb:{ok_e}/{len(result)}", flush=True)


if __name__ == "__main__":
    main()
