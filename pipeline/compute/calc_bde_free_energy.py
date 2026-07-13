#!/usr/bin/env python
"""BOND DISSOCIATION FREE ENERGY (BDFE) pilot -- proper thermal correction, not the raw
electronic-energy BDE in calc_bde.py.

calc_bde.py computed BDE = E_el(fragA) + E_el(fragB) - E_el(parent) with NO ZPE/thermal
correction (plain --opt / single-point). This is why the aldehyde C-H values (100-117
kcal/mol) ran well above the experimental ballpark (~88 kcal/mol for a simple aldehyde):
breaking a bond changes vibrational/rotational/translational degrees of freedom, so
Delta-E != Delta-H != Delta-G, and the gap is dominated by ZPE (which is large and negative
for a light X-H bond).

This script instead uses xtb --ohess (full Hessian -> vibrational frequencies -> ZPE +
thermal + entropy corrections -> G_RRHO), reusing thermo_orca.run_ohess's g98.out output for
an OPTIONAL Shermo quasi-RRHO re-correction (same tool this project already uses for the
main Delta-G modeling target, see thermo_orca.py) -- avoids the standard RRHO's known
overestimated entropy for near-zero-frequency modes.

The isolated H atom (aldehyde-side fragment) has NO vibrational or rotational degrees of
freedom -- running --ohess on a single atom is meaningless. Its G is computed analytically
(ideal monatomic gas: Sackur-Tetrode translational entropy + electronic degeneracy g=2 for
the doublet ground state), a standard textbook formula, not from xtb/Shermo.

Reports, per molecule: BDE_E (raw electronic, comparable to calc_bde.py), BDE_G_xtb (xtb's
own RRHO free energy), BDE_G_shermo (Shermo qRRHO, if available) -- lets us see exactly how
much ZPE/thermal correction shifts the number.

PILOT usage:
  python calc_bde_free_energy.py --which aldehydes --n 15 --out /tmp/bdfe_pilot_ald.csv
  python calc_bde_free_energy.py --which products --n 15 --out /tmp/bdfe_pilot_prod.csv
"""
import argparse
import math
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

RDLogger.DisableLog("rdApp.*")
HARTREE_TO_KCAL = 627.509474

# ── physical constants (SI), for the analytic H-atom ideal-gas correction ──
_H_PLANCK = 6.62607015e-34      # J*s
_K_B = 1.380649e-23             # J/K
_N_A = 6.02214076e23            # 1/mol
_R_GAS = 8.31446262              # J/(mol*K)
_M_H_KG = 1.00782503e-3 / _N_A  # kg, mass of one H atom
_HARTREE_J = 4.3597447222071e-18


def g_h_atom_correction(T: float = 298.15, P_atm: float = 1.0) -> float:
    """G(T,P) - E_el for an isolated ground-state H atom (Eh): ideal monatomic gas
    translational (Sackur-Tetrode) + PV term + electronic degeneracy (doublet, g=2), NO
    vibrational/rotational contribution (a single atom has none). Standard textbook result,
    independent of any xtb/Shermo calculation."""
    P_pa = P_atm * 101325.0
    V = _R_GAS * T / (P_pa * _N_A)  # volume per molecule, m^3 (ideal gas, per-particle)
    q_trans = ((2 * math.pi * _M_H_KG * _K_B * T) / _H_PLANCK**2) ** 1.5 * V
    S_trans = _K_B * (math.log(q_trans) + 2.5)            # J/K, per atom
    S_elec = _K_B * math.log(2.0)                          # doublet ground state degeneracy
    S = (S_trans + S_elec) * _N_A                          # J/(mol*K)
    H_corr = 2.5 * _R_GAS * T                               # (3/2 RT trans U) + (RT, PV term)
    G_corr_J_per_mol = H_corr - T * S
    return G_corr_J_per_mol / (_HARTREE_J * _N_A)           # -> Eh


H_ATOM_G_CORR = g_h_atom_correction()  # ~ -0.0087 Eh at 298.15K/1atm -- computed once


def _xyz_block(symbols, coords) -> str:
    lines = [str(len(symbols)), ""]
    for s, c in zip(symbols, coords):
        lines.append(f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}")
    return "\n".join(lines) + "\n"


def _run_ohess_uhf(xyz_str, work_dir, xtb_bin, charge, uhf, T=298.15, P_atm=1.0, timeout=1800):
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "mol.xyz").write_text(xyz_str, encoding="utf-8")
    Th._write_xtb_inp(work_dir / "xtb.inp", T, P_atm * 1.01325)
    cmd = [xtb_bin, "mol.xyz", "--gfn", "2", "--ohess", "tight", "--input", "xtb.inp",
          "--chrg", str(charge), "--uhf", str(uhf), "--norestart", "--parallel", "1"]
    stdout = ""
    try:
        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout)
        stdout = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        pass
    g98 = work_dir / "g98.out"
    return stdout, (g98 if g98.exists() else None)


def mol_with_bonds(xyz_file: str):
    mol = Chem.MolFromXYZFile(xyz_file)
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None
    return mol


def split_at_bond(mol, i, j, coords, symbols):
    em = Chem.RWMol(mol)
    if em.GetBondBetweenAtoms(i, j) is None:
        return None
    em.RemoveBond(i, j)
    frags = Chem.GetMolFrags(em, asMols=False, sanitizeFrags=False)
    fa = next(f for f in frags if i in f)
    fb = next(f for f in frags if j in f)
    if fa is fb:
        return None
    return [([symbols[k] for k in idx], coords[list(idx)]) for idx in (fa, fb)]


def _frag_G(symbols, coords, work_dir, xtb_bin, shermo_bin, uhf: int) -> dict:
    """G_E (electronic only), G_xtb (RRHO), G_shermo (qRRHO, or None).
    uhf=0 for the closed-shell parent, uhf=1 for each homolysis radical fragment."""
    if len(symbols) == 1:  # the isolated H atom -- analytic correction, no ohess
        e = None
        try:
            wd = work_dir / "h_sp"; wd.mkdir(parents=True, exist_ok=True)
            (wd / "h.xyz").write_text(_xyz_block(symbols, coords))
            r = subprocess.run([xtb_bin, "h.xyz", "--gfn", "2", "--chrg", "0", "--uhf", str(uhf),
                               "--norestart", "--parallel", "1"],
                              cwd=str(wd), capture_output=True, text=True, timeout=60)
            e = A._parse_xtb_energy(r.stdout)
        except Exception:
            pass
        if e is None:
            return {"E": None, "G_xtb": None, "G_shermo": None}
        return {"E": e, "G_xtb": e + H_ATOM_G_CORR, "G_shermo": e + H_ATOM_G_CORR}

    stdout, g98 = _run_ohess_uhf(_xyz_block(symbols, coords), work_dir, xtb_bin, charge=0, uhf=uhf)
    E_el = Th._parse_xtb_energy(stdout)
    G_xtb = Th.parse_xtb_G(stdout)
    G_sh = None
    if shermo_bin and g98:
        out = Th.run_shermo(g98, shermo_bin, E_el=E_el)
        G_sh = Th.parse_shermo_G(out)
    return {"E": E_el, "G_xtb": G_xtb, "G_shermo": G_sh}


def bdfe_aldehyde_ch(xyz_file, xtb_bin, shermo_bin, work_dir) -> dict:
    row = {k: np.nan for k in ("bde_E_kcal", "bdfe_xtb_kcal", "bdfe_shermo_kcal")}
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

    parent = _frag_G(symbols, coords, work_dir / "parent", xtb_bin, shermo_bin, uhf=0)
    fragA = _frag_G(symA, coordA, work_dir / "fragA", xtb_bin, shermo_bin, uhf=1)
    fragB = _frag_G(symB, coordB, work_dir / "fragB", xtb_bin, shermo_bin, uhf=1)
    if None not in (parent["E"], fragA["E"], fragB["E"]):
        row["bde_E_kcal"] = round((fragA["E"] + fragB["E"] - parent["E"]) * HARTREE_TO_KCAL, 3)
    if None not in (parent["G_xtb"], fragA["G_xtb"], fragB["G_xtb"]):
        row["bdfe_xtb_kcal"] = round((fragA["G_xtb"] + fragB["G_xtb"] - parent["G_xtb"]) * HARTREE_TO_KCAL, 3)
    if None not in (parent["G_shermo"], fragA["G_shermo"], fragB["G_shermo"]):
        row["bdfe_shermo_kcal"] = round((fragA["G_shermo"] + fragB["G_shermo"] - parent["G_shermo"]) * HARTREE_TO_KCAL, 3)
    return row


def bdfe_product_cc(xyz_file, xtb_bin, shermo_bin, work_dir) -> dict:
    row = {k: np.nan for k in ("bde_E_kcal", "bdfe_xtb_kcal", "bdfe_shermo_kcal")}
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

    parent = _frag_G(symbols, coords, work_dir / "parent", xtb_bin, shermo_bin, uhf=0)
    fragA = _frag_G(symA, coordA, work_dir / "fragA", xtb_bin, shermo_bin, uhf=1)
    fragB = _frag_G(symB, coordB, work_dir / "fragB", xtb_bin, shermo_bin, uhf=1)
    if None not in (parent["E"], fragA["E"], fragB["E"]):
        row["bde_E_kcal"] = round((fragA["E"] + fragB["E"] - parent["E"]) * HARTREE_TO_KCAL, 3)
    if None not in (parent["G_xtb"], fragA["G_xtb"], fragB["G_xtb"]):
        row["bdfe_xtb_kcal"] = round((fragA["G_xtb"] + fragB["G_xtb"] - parent["G_xtb"]) * HARTREE_TO_KCAL, 3)
    if None not in (parent["G_shermo"], fragA["G_shermo"], fragB["G_shermo"]):
        row["bdfe_shermo_kcal"] = round((fragA["G_shermo"] + fragB["G_shermo"] - parent["G_shermo"]) * HARTREE_TO_KCAL, 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--n", type=int, default=None, help="pilot mode: first N molecules")
    ap.add_argument("--chunk-id", type=int, default=None, help="array mode: which chunk")
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--out", default=None, help="pilot mode: single output CSV")
    ap.add_argument("--out-dir", default=None, help="array mode: chunk_NNNN.csv written here")
    ap.add_argument("--xtb-bin", default="/home/schen3/xtb/bin/xtb")
    ap.add_argument("--shermo-bin", default="/home/schen3/.local/bin/Shermo")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    shermo_bin = args.shermo_bin if Path(args.shermo_bin).exists() else None
    print(f"H atom G correction (analytic): {H_ATOM_G_CORR:.6f} Eh "
         f"({H_ATOM_G_CORR * HARTREE_TO_KCAL:.3f} kcal/mol)  shermo={'yes' if shermo_bin else 'NO'}",
         flush=True)

    H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
    src = H / f"{args.which}_all.csv"
    df = pd.read_csv(src, usecols=["id", "xyz_file", "error"], dtype=str,
                     keep_default_na=False, low_memory=False)
    df = df[(df["error"] == "") & (df["xyz_file"] != "")].drop_duplicates("id").reset_index(drop=True)

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
        print(f"BDFE {args.which} chunk {args.chunk_id}: rows {lo}:{hi} ({len(df)})", flush=True)
    else:
        df = df.head(args.n or 15)
        out_path = Path(args.out)
        print(f"BDFE pilot on {len(df)} {args.which}", flush=True)

    wd_root = Path(args.work_dir or "/tmp/bdfe_pilot")
    rows = []
    for _, rec in df.iterrows():
        wd = wd_root / f"m{rec['id']}"
        try:
            if args.which == "aldehydes":
                r = bdfe_aldehyde_ch(rec["xyz_file"], args.xtb_bin, shermo_bin, wd)
            else:
                r = bdfe_product_cc(rec["xyz_file"], args.xtb_bin, shermo_bin, wd)
        except Exception as e:
            r = {"error": str(e)}
        r["id"] = rec["id"]
        rows.append(r)
        print(rec["id"], r, flush=True)
        shutil.rmtree(wd, ignore_errors=True)

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    ok_e = result["bde_E_kcal"].notna().sum() if "bde_E_kcal" in result else 0
    ok_g = result["bdfe_xtb_kcal"].notna().sum() if "bdfe_xtb_kcal" in result else 0
    ok_sh = result["bdfe_shermo_kcal"].notna().sum() if "bdfe_shermo_kcal" in result else 0
    print(f"wrote {out_path}  E:{ok_e}/{len(result)}  G_xtb:{ok_g}/{len(result)}  G_shermo:{ok_sh}/{len(result)}", flush=True)


if __name__ == "__main__":
    main()
