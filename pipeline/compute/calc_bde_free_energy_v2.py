#!/usr/bin/env python
"""BDFE v2 -- OPTIMIZED: reuses the parent molecule's G_xtb/xtb_energy (already computed
during the original funnel_v3 featurization, stored in {aldehydes,products}_all.csv) instead
of re-running --ohess on the parent. Only the two homolysis radical FRAGMENTS (new species
that were never computed before) get a fresh --ohess. Roughly halves (aldehyde) to a third
(product) of the xtb cost vs calc_bde_free_energy.py v1, which wastefully recomputed the
parent from scratch.

Shermo qRRHO is DROPPED in this version: v1's pilot (20 mols) showed Shermo agrees with
xtb's own RRHO to <0.5 kcal/mol on this system (no severe low-frequency-mode pathology), and
there's no cheap way to get a Shermo-consistent PARENT value (its g98.out was deleted right
after the original featurization run, per this project's per-molecule scratch-cleanup
discipline) -- recomputing it would defeat the whole point of this optimization. Only
BDFE_xtb (self-consistent RRHO throughout: reused parent + fresh fragments) is reported.

PILOT usage (validates the reuse logic before any array submission):
  python calc_bde_free_energy_v2.py --which aldehydes --n 15 --out /tmp/bdfe2_pilot_ald.csv
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

_H_PLANCK = 6.62607015e-34
_K_B = 1.380649e-23
_N_A = 6.02214076e23
_R_GAS = 8.31446262
_M_H_KG = 1.00782503e-3 / _N_A
_HARTREE_J = 4.3597447222071e-18


def g_h_atom_correction(T: float = 298.15, P_atm: float = 1.0) -> float:
    P_pa = P_atm * 101325.0
    V = _R_GAS * T / (P_pa * _N_A)
    q_trans = ((2 * math.pi * _M_H_KG * _K_B * T) / _H_PLANCK**2) ** 1.5 * V
    S_trans = _K_B * (math.log(q_trans) + 2.5)
    S_elec = _K_B * math.log(2.0)
    S = (S_trans + S_elec) * _N_A
    H_corr = 2.5 * _R_GAS * T
    return (H_corr - T * S) / (_HARTREE_J * _N_A)


H_ATOM_G_CORR = g_h_atom_correction()


def _xyz_block(symbols, coords) -> str:
    lines = [str(len(symbols)), ""]
    for s, c in zip(symbols, coords):
        lines.append(f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}")
    return "\n".join(lines) + "\n"


def _run_ohess_uhf(xyz_str, work_dir, xtb_bin, charge, uhf, solvent="dmso", T=298.15, P_atm=1.0, timeout=1800):
    """solvent="dmso" by default: MUST match the parent's G_xtb, which cb_featurize.py
    computed with --alpb dmso (its --solvent default). Mixing a solvated parent with
    gas-phase fragments is not just imprecise, it's a real methodological inconsistency --
    caught via a v1-vs-v2 discrepancy (~4.6 kcal/mol) on the same molecules that turned out
    to be exactly this, not numerical noise."""
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


def _frag_G_E(symbols, coords, work_dir, xtb_bin, solvent="dmso") -> dict:
    """Fresh calc for a RADICAL fragment (uhf=1 always -- this function is only ever
    called on homolysis fragments, never the parent, which is looked up instead).
    solvent="dmso" to match the parent's methodology (see _run_ohess_uhf docstring)."""
    if len(symbols) == 1:  # isolated H atom -- analytic correction, no ohess possible.
        # E_el computed WITH --alpb dmso (matches the parent/fragA electronic-energy
        # treatment); the added thermal correction stays the gas-phase ideal-monatomic-gas
        # formula regardless -- solvent's effect on entropy/ZPE is far smaller than on
        # electronic energy, and there's no vibrational Hessian to solvent-correct for a
        # single atom anyway. Standard approximation (solvated E_el + gas-phase G_thermal).
        e = None
        try:
            wd = work_dir / "h_sp"; wd.mkdir(parents=True, exist_ok=True)
            (wd / "h.xyz").write_text(_xyz_block(symbols, coords))
            cmd = [xtb_bin, "h.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "1",
                  "--norestart", "--parallel", "1"]
            if solvent:
                cmd += ["--alpb", solvent]
            r = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True, timeout=60)
            e = A._parse_xtb_energy(r.stdout)
        except Exception:
            pass
        if e is None:
            return {"E": None, "G_xtb": None}
        return {"E": e, "G_xtb": e + H_ATOM_G_CORR}

    stdout = _run_ohess_uhf(_xyz_block(symbols, coords), work_dir, xtb_bin, charge=0, uhf=1, solvent=solvent)
    return {"E": Th._parse_xtb_energy(stdout), "G_xtb": Th.parse_xtb_G(stdout)}


def bdfe_aldehyde_ch(xyz_file, parent_E, parent_G, xtb_bin, work_dir) -> dict:
    row = {"bde_E_kcal": np.nan, "bdfe_xtb_kcal": np.nan}
    if parent_E is None or parent_G is None:
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

    fragA = _frag_G_E(symA, coordA, work_dir / "fragA", xtb_bin)
    fragB = _frag_G_E(symB, coordB, work_dir / "fragB", xtb_bin)
    if None not in (fragA["E"], fragB["E"]):
        row["bde_E_kcal"] = round((fragA["E"] + fragB["E"] - parent_E) * HARTREE_TO_KCAL, 3)
    if None not in (fragA["G_xtb"], fragB["G_xtb"]):
        row["bdfe_xtb_kcal"] = round((fragA["G_xtb"] + fragB["G_xtb"] - parent_G) * HARTREE_TO_KCAL, 3)
    return row


def bdfe_product_cc(xyz_file, parent_E, parent_G, xtb_bin, work_dir) -> dict:
    row = {"bde_E_kcal": np.nan, "bdfe_xtb_kcal": np.nan}
    if parent_E is None or parent_G is None:
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

    fragA = _frag_G_E(symA, coordA, work_dir / "fragA", xtb_bin)
    fragB = _frag_G_E(symB, coordB, work_dir / "fragB", xtb_bin)
    if None not in (fragA["E"], fragB["E"]):
        row["bde_E_kcal"] = round((fragA["E"] + fragB["E"] - parent_E) * HARTREE_TO_KCAL, 3)
    if None not in (fragA["G_xtb"], fragB["G_xtb"]):
        row["bdfe_xtb_kcal"] = round((fragA["G_xtb"] + fragB["G_xtb"] - parent_G) * HARTREE_TO_KCAL, 3)
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
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    print(f"H atom G correction (analytic): {H_ATOM_G_CORR:.6f} Eh "
         f"({H_ATOM_G_CORR * HARTREE_TO_KCAL:.3f} kcal/mol)", flush=True)

    H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
    src = H / f"{args.which}_all.csv"
    df = pd.read_csv(src, usecols=["id", "xyz_file", "error", "xtb_energy", "G_xtb"], dtype=str,
                     keep_default_na=False, low_memory=False)
    df = df[(df["error"] == "") & (df["xyz_file"] != "") & (df["xtb_energy"] != "") & (df["G_xtb"] != "")]
    df = df.drop_duplicates("id").reset_index(drop=True)
    df["xtb_energy"] = df["xtb_energy"].astype(float)
    df["G_xtb"] = df["G_xtb"].astype(float)

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
        print(f"BDFEv2 {args.which} chunk {args.chunk_id}: rows {lo}:{hi} ({len(df)})", flush=True)
    else:
        df = df.head(args.n or 15)
        out_path = Path(args.out)
        print(f"BDFEv2 pilot on {len(df)} {args.which}", flush=True)

    wd_root = Path(args.work_dir or "/tmp/bdfe2_pilot")
    rows = []
    for _, rec in df.iterrows():
        wd = wd_root / f"m{rec['id']}"
        try:
            if args.which == "aldehydes":
                r = bdfe_aldehyde_ch(rec["xyz_file"], rec["xtb_energy"], rec["G_xtb"], args.xtb_bin, wd)
            else:
                r = bdfe_product_cc(rec["xyz_file"], rec["xtb_energy"], rec["G_xtb"], args.xtb_bin, wd)
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
    print(f"wrote {out_path}  E:{ok_e}/{len(result)}  G_xtb:{ok_g}/{len(result)}", flush=True)


if __name__ == "__main__":
    main()
