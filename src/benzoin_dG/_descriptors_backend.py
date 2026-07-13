#!/usr/bin/env python3
"""
Aldehyde Descriptor Calculator
================================
Pipeline per aldehyde SMILES:
  Level 0  (RDKit only)   : 2D descriptors, Gasteiger charges
  Level 1  (xTB CLI)      : GFN2-xTB geometry opt, HOMO/LUMO/gap, IP/EA,
                             global electrophilicity, dipole, Mulliken charges,
                             Fukui f+/f-/f0 at carbonyl carbon
  Level 2  (morfeus-ml)   : %Vbur, Sterimol L/B1/B5, SASA,
                             Dispersion P_int
  Level 3  (Multiwfn)     : GFN2 wfn HOMO/LUMO/gap, ADCH charges + Fukui, QTAIM C=O BCP

Install dependencies
--------------------
  pip install rdkit morfeus-ml
  # xtb (pick one):
  conda install -c conda-forge xtb                # recommended
  # or download binary from https://github.com/grimme-lab/xtb/releases
  # Multiwfn: http://sobereva.com/multiwfn/

Usage
-----
  python ald_descriptors.py                        # reads aldehydes.csv
  python ald_descriptors.py --max 100              # first 100 rows
  python ald_descriptors.py --smiles "O=Cc1ccccc1"
  python ald_descriptors.py --xtb-bin /path/to/xtb
  python ald_descriptors.py --multiwfn-bin /path/to/Multiwfn --multiwfn
  python ald_descriptors.py --no-xtb-opt           # skip opt, use MMFF only

Output: ald_desc/descriptors.csv   (one row per aldehyde, streamed)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ── Optional imports ───────────────────────────────────────────────────────
try:
    from morfeus import BuriedVolume, Sterimol, SASA
    try:
        from morfeus import Dispersion
        HAS_DISP = True
    except ImportError:
        HAS_DISP = False
    HAS_MORFEUS = True
except ImportError:
    HAS_MORFEUS = False
    HAS_DISP = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VBUR_RADIUS = 3.5     # Å, standard for organocatalysis
CO_BOND_MAX = 1.28    # Å  C=O double bond cutoff (single bond > 1.30)
CH_BOND_MAX = 1.15    # Å  C-H bond cutoff


# ══════════════════════════════════════════════════════════════════════════
#  3D embedding
# ══════════════════════════════════════════════════════════════════════════

def smiles_to_3d(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    params.maxIterations = 1000
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return None
    if AllChem.MMFFOptimizeMolecule(mol, maxIters=2000) == -1:
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def smiles_to_3d_multi(smiles: str, n_confs: int, title: str = "") -> list[str]:
    """
    Generate n_confs ETKDG+MMFF conformers; return list of XYZ strings.
    Each entry is an independent starting geometry for xTB optimization.
    Falls back to a single ETKDG conformer if EmbedMultipleConfs fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    params.maxIterations = 1000

    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params))
    if not cids:
        single = smiles_to_3d(smiles)
        return [mol_to_xyz_str(single, f"{title}_c0")] if single else []

    AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=2000)

    result = []
    for i, cid in enumerate(cids):
        conf = mol_h.GetConformer(cid)
        n = mol_h.GetNumAtoms()
        lines = [str(n), f"{title}_c{i}"]
        for atom in mol_h.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol():<3s} {p.x:14.8f} {p.y:14.8f} {p.z:14.8f}")
        result.append("\n".join(lines) + "\n")
    return result


def mol_to_xyz_str(mol: Chem.Mol, title: str = "") -> str:
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), title]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<3s} {p.x:14.8f} {p.y:14.8f} {p.z:14.8f}")
    return "\n".join(lines) + "\n"


def parse_xyz(xyz_str: str) -> tuple[list[str], np.ndarray]:
    lines = xyz_str.strip().splitlines()
    n = int(lines[0])
    symbols, coords = [], []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return symbols, np.array(coords)


# ══════════════════════════════════════════════════════════════════════════
#  Aldehyde atom finder (works on XYZ with explicit H)
# ══════════════════════════════════════════════════════════════════════════

def find_aldehyde_atoms(
    symbols: list[str], coords: np.ndarray
) -> list[tuple[int, int, int]]:
    """
    Return list of (c_idx, o_idx, r_idx) for each -CHO group.
    All indices are 0-based.
    r_idx = heavy atom on R side of CHO (-1 if not found).
    """
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    n = len(symbols)
    hits = []

    for i, sym in enumerate(symbols):
        if sym != "C":
            continue
        o_nbrs = [j for j in range(n)
                  if j != i and symbols[j] == "O" and dist[i, j] < CO_BOND_MAX]
        h_nbrs = [j for j in range(n)
                  if j != i and symbols[j] == "H" and dist[i, j] < CH_BOND_MAX]
        if len(o_nbrs) == 1 and len(h_nbrs) >= 1:
            o = o_nbrs[0]
            # R atom: heavy neighbor that is not O
            r = next(
                (j for j in range(n)
                 if j != i and j != o and symbols[j] != "H" and dist[i, j] < 1.75),
                -1,
            )
            hits.append((i, o, r))
    return hits


# ══════════════════════════════════════════════════════════════════════════
#  Level 0 — RDKit 2D descriptors
# ══════════════════════════════════════════════════════════════════════════

_RDKIT_CALC = [
    ("MW",              Descriptors.MolWt),
    ("ExactMW",         Descriptors.ExactMolWt),
    ("LogP",            Descriptors.MolLogP),
    ("MolMR",           Descriptors.MolMR),
    ("TPSA",            Descriptors.TPSA),
    ("HBD",             rdMolDescriptors.CalcNumHBD),
    ("HBA",             rdMolDescriptors.CalcNumHBA),
    ("RotBonds",        rdMolDescriptors.CalcNumRotatableBonds),
    ("ArRings",         rdMolDescriptors.CalcNumAromaticRings),
    ("ArHetRings",      rdMolDescriptors.CalcNumAromaticHeterocycles),
    ("AlRings",         rdMolDescriptors.CalcNumAliphaticRings),
    ("Rings",           rdMolDescriptors.CalcNumRings),
    ("Heteroatoms",     rdMolDescriptors.CalcNumHeteroatoms),
    ("FractionCSP3",    rdMolDescriptors.CalcFractionCSP3),
    ("BertzCT",         Descriptors.BertzCT),
    ("Chi0v",           Descriptors.Chi0v),
    ("Chi1v",           Descriptors.Chi1v),
    ("Kappa1",          Descriptors.Kappa1),
    ("Kappa2",          Descriptors.Kappa2),
    ("LabuteASA",       rdMolDescriptors.CalcLabuteASA),
    ("NumStereocenters",rdMolDescriptors.CalcNumAtomStereoCenters),
]

_RDKIT_FIELDS = [n for n, _ in _RDKIT_CALC] + [
    "n_CHO", "gasteiger_CHO_C", "gasteiger_CHO_O",
    "gasteiger_maxpos", "gasteiger_minneg",
]


def calc_rdkit(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {f: None for f in _RDKIT_FIELDS}

    desc: dict[str, Any] = {}
    for name, fn in _RDKIT_CALC:
        try:
            desc[name] = round(fn(mol), 6)
        except Exception:
            desc[name] = None

    # Gasteiger charges
    mol_g = Chem.RWMol(Chem.AddHs(mol))
    try:
        ComputeGasteigerCharges(mol_g)
        charges = [a.GetDoubleProp("_GasteigerCharge") for a in mol_g.GetAtoms()]
        desc["gasteiger_maxpos"] = round(max(charges), 6)
        desc["gasteiger_minneg"] = round(min(charges), 6)

        pattern = Chem.MolFromSmarts("[CX3H1](=[O])")
        matches = mol.GetSubstructMatches(pattern)
        desc["n_CHO"] = len(matches)

        if matches:
            # Use the mol with H to get charges at the correct atoms
            mol_h = Chem.AddHs(mol)
            ComputeGasteigerCharges(mol_h)
            m = mol.GetSubstructMatches(pattern)[0]
            c_idx, o_idx = m[0], m[1]
            desc["gasteiger_CHO_C"] = round(
                mol_h.GetAtomWithIdx(c_idx).GetDoubleProp("_GasteigerCharge"), 6)
            desc["gasteiger_CHO_O"] = round(
                mol_h.GetAtomWithIdx(o_idx).GetDoubleProp("_GasteigerCharge"), 6)
        else:
            desc["gasteiger_CHO_C"] = desc["gasteiger_CHO_O"] = None
    except Exception:
        desc.setdefault("n_CHO", None)
        desc.setdefault("gasteiger_CHO_C", None)
        desc.setdefault("gasteiger_CHO_O", None)
        desc.setdefault("gasteiger_maxpos", None)
        desc.setdefault("gasteiger_minneg", None)

    return desc


# ══════════════════════════════════════════════════════════════════════════
#  Level 1 — xTB via CLI
# ══════════════════════════════════════════════════════════════════════════

_XTB_FIELDS = [
    "xtb_energy", "xtb_HOMO", "xtb_LUMO", "xtb_gap",
    "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta", "xtb_omega", "xtb_dipole",
    "mulliken_CHO_C", "mulliken_CHO_O",
    "fukui_plus_CHO_C", "fukui_minus_CHO_C", "fukui_0_CHO_C",
    "dual_descriptor_CHO_C",
    "wbo_CO", "xtb_alpha0", "xtb_C6",
    "pa_CHO_O",
]


def _run_xtb(args: list[str], work_dir: Path, timeout: int = 300) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            args, cwd=work_dir, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        log.warning("xtb timed out: %s", " ".join(args))
    except Exception as e:
        log.warning("xtb error: %s", e)
    return None


def xtb_optimize(xyz_str: str, work_dir: Path, xtb_bin: str,
                 charge: int = 0) -> str | None:
    """GFN2-xTB geometry optimization; returns optimized XYZ string."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "input.xyz").write_text(xyz_str, encoding="utf-8")
    r = _run_xtb(
        [xtb_bin, "input.xyz", "--opt", "tight", "--gfn", "2",
         "--charge", str(charge), "--norestart", "--parallel", "1"],
        work_dir,
    )
    opt = work_dir / "xtbopt.xyz"
    if opt.exists():
        return opt.read_text(encoding="utf-8")
    if r:
        log.debug("xtb opt stdout: %s", r.stdout[-500:])
    return None


def _xtb_opt_energy(xyz_str: str, work_dir: Path,
                    xtb_bin: str, charge: int = 0) -> tuple[str | None, float | None]:
    """xTB --opt tight; return (optimized_xyz_str, energy_Eh) or (None, None)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "input.xyz").write_text(xyz_str, encoding="utf-8")
    r = _run_xtb(
        [xtb_bin, "input.xyz", "--opt", "tight", "--gfn", "2",
         "--charge", str(charge), "--norestart", "--parallel", "1"],
        work_dir,
    )
    opt = work_dir / "xtbopt.xyz"
    if not opt.exists():
        return None, None
    energy = _parse_xtb_energy(r.stdout) if r else None
    return opt.read_text(encoding="utf-8"), energy


def pick_best_xtb_conformer(
    xyz_strings: list[str], work_dir: Path, xtb_bin: str
) -> tuple[str, bool]:
    """
    xTB-optimize every conformer; return (xyz of lowest-energy conformer, optimized_flag).
    Conformer subdirs are named c000, c001, … inside work_dir.
    """
    best_xyz = xyz_strings[0]
    best_E   = float("inf")
    any_opt  = False

    for i, xyz in enumerate(xyz_strings):
        opt_xyz, energy = _xtb_opt_energy(xyz, work_dir / f"c{i:03d}", xtb_bin)
        if opt_xyz is None:
            continue
        any_opt = True
        if energy is not None and energy < best_E:
            best_E   = energy
            best_xyz = opt_xyz
        elif energy is None and best_E == float("inf"):
            best_xyz = opt_xyz   # take first valid when energies unavailable

    return best_xyz, any_opt


def _parse_xtb_energy(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if "TOTAL ENERGY" in line:
            try:
                return float(line.split()[3])
            except (IndexError, ValueError):
                pass
    return None


def _parse_xtb_homo_lumo(stdout: str) -> tuple[float | None, float | None]:
    """Return HOMO/LUMO of the NEUTRAL molecule (first occurrence in output)."""
    homo = lumo = None
    for line in stdout.splitlines():
        if "(HOMO)" in line and homo is None:
            try:
                homo = float(line.split()[-2])
            except (IndexError, ValueError):
                pass
        if "(LUMO)" in line and lumo is None:
            try:
                lumo = float(line.split()[-2])
            except (IndexError, ValueError):
                pass
        if homo is not None and lumo is not None:
            break
    return homo, lumo


def _parse_xtb_dipole(stdout: str) -> float | None:
    """Extract total dipole (Debye) from first 'full:' line after 'molecular dipole'."""
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if "molecular dipole:" in line.lower():
            for bl in lines[i : i + 6]:
                if "full:" in bl.lower():
                    try:
                        return float(bl.split()[-1])
                    except (IndexError, ValueError):
                        pass
            break  # only first dipole block (neutral molecule)
    return None


def _parse_xtb_alpha_c6(stdout: str) -> tuple[float | None, float | None]:
    """Extract molecular α(0) (au) and C6AA (au·bohr⁶) from xTB stdout."""
    alpha = c6 = None
    for line in stdout.splitlines():
        if "Mol. C6AA" in line:
            try:
                c6 = float(line.split()[-1])
            except (IndexError, ValueError):
                pass
        if "Mol. α(0)" in line or "Mol. alpha(0)" in line:
            try:
                alpha = float(line.split()[-1])
            except (IndexError, ValueError):
                pass
        if alpha is not None and c6 is not None:
            break
    return alpha, c6


def _parse_wbo(wbo_file: Path, c_idx: int, o_idx: int) -> float | None:
    """Return Wiberg bond order for the C–O pair (0-based indices) from xTB wbo file."""
    if not wbo_file.exists():
        return None
    c1, o1 = c_idx + 1, o_idx + 1   # wbo file uses 1-based indexing
    try:
        for line in wbo_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                i, j = int(parts[0]), int(parts[1])
                if (i == c1 and j == o1) or (i == o1 and j == c1):
                    return round(float(parts[2]), 6)
    except Exception:
        pass
    return None


def _calc_proton_affinity(
    xyz_str: str, symbols: list[str], coords: np.ndarray,
    xtb_bin: str, work_dir: Path, e_neutral: float | None,
) -> float | None:
    """
    Proton affinity at carbonyl O: PA = E(neutral) - E(protonated) in kcal/mol.
    H is placed along the C→O vector at 0.97 Å from O; protonated species optimized
    with charge=+1.  Positive PA means thermodynamically favorable protonation.
    """
    if e_neutral is None:
        return None
    ald_atoms = find_aldehyde_atoms(symbols, coords)
    if not ald_atoms:
        return None
    c_idx, o_idx, _ = ald_atoms[0]
    o_pos = coords[o_idx]
    c_pos = coords[c_idx]
    co_unit = (o_pos - c_pos) / np.linalg.norm(o_pos - c_pos)
    h_pos = o_pos + co_unit * 0.97  # 0.97 Å O–H bond
    new_sym = list(symbols) + ["H"]
    new_xyz = np.vstack([coords, h_pos])
    n = len(new_sym)
    xyz_lines = [str(n), "protonated_O"]
    for s, p in zip(new_sym, new_xyz):
        xyz_lines.append(f"{s}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}")
    prot_xyz = "\n".join(xyz_lines)
    _, e_prot = _xtb_opt_energy(prot_xyz, work_dir / "prot_O", xtb_bin, charge=1)
    if e_prot is None:
        # fall back: SP on initial protonated geometry
        r = _xtb_sp(prot_xyz, work_dir / "prot_O_sp", xtb_bin, charge=1)
        if r is None:
            return None
        e_prot = _parse_xtb_energy(r.stdout)
    if e_prot is None:
        return None
    return round((e_neutral - e_prot) * 627.509, 4)


def _parse_xtb_charges(charges_file: Path) -> list[float] | None:
    if not charges_file.exists():
        return None
    try:
        return [float(l.strip()) for l in charges_file.read_text().splitlines() if l.strip()]
    except Exception:
        return None


def _xtb_sp(xyz_str: str, work_dir: Path, xtb_bin: str,
             charge: int = 0, extra_flags: list[str] | None = None) -> subprocess.CompletedProcess | None:
    """Run a GFN2-xTB single-point; return CompletedProcess or None."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "mol.xyz").write_text(xyz_str, encoding="utf-8")
    flags = extra_flags or []
    return _run_xtb(
        [xtb_bin, "mol.xyz", "--gfn", "2", "--charge", str(charge),
         "--norestart"] + flags,
        work_dir,
    )


def _fukui_finite_diff(
    xyz_str: str, symbols: list[str], coords: np.ndarray,
    xtb_bin: str, work_dir: Path
) -> dict[str, list[float]] | None:
    """
    Condensed Fukui functions via finite difference on Mulliken charges:
      f+(i) = q(N+1)_i - q(N)_i   ← electrophilicity (susceptibility to nucleophilic attack)
      f-(i) = q(N)_i   - q(N-1)_i ← nucleophilicity  (susceptibility to electrophilic attack)
      f0(i) = (f+ + f-) / 2        ← radical
    """
    n_atoms = len(symbols)

    def get_charges(chrg: int, label: str) -> list[float] | None:
        d = work_dir / label
        r = _xtb_sp(xyz_str, d, xtb_bin, charge=chrg)
        return _parse_xtb_charges(d / "charges") if r else None

    q_n   = get_charges(0,  "neutral")
    q_np1 = get_charges(-1, "anion")    # N+1 electrons = charge -1
    q_nm1 = get_charges(+1, "cation")   # N-1 electrons = charge +1

    if q_n is None or len(q_n) != n_atoms:
        return None

    # f+(i) = q(N)_i - q(N+1)_i  [charge decreases = electrons gained = electrophilic site]
    # f-(i) = q(N-1)_i - q(N)_i  [charge increases = electrons lost = nucleophilic site]
    f_plus  = [round((q_n[i] - q_np1[i]), 6) if q_np1 and i < len(q_np1) else None
               for i in range(n_atoms)]
    f_minus = [round((q_nm1[i] - q_n[i]), 6) if q_nm1 and i < len(q_nm1) else None
               for i in range(n_atoms)]
    f_zero  = [round((f_plus[i] + f_minus[i]) / 2, 6)
               if f_plus[i] is not None and f_minus[i] is not None else None
               for i in range(n_atoms)]

    return {"+": f_plus, "-": f_minus, "0": f_zero, "q_neutral": q_n}


def calc_xtb(
    xyz_str: str,
    symbols: list[str],
    coords: np.ndarray,
    xtb_bin: str,
    work_dir: Path,
    charge: int = 0,
) -> dict[str, Any]:
    desc: dict[str, Any] = {f: None for f in _XTB_FIELDS}

    # ── Neutral single-point (energy, HOMO/LUMO, dipole, charges) ─────
    r_sp = _xtb_sp(xyz_str, work_dir / "neutral", xtb_bin, charge)
    if r_sp is None:
        return desc
    stdout = r_sp.stdout
    desc["xtb_energy"] = _parse_xtb_energy(stdout)
    homo, lumo = _parse_xtb_homo_lumo(stdout)
    desc["xtb_HOMO"]   = homo
    desc["xtb_LUMO"]   = lumo
    desc["xtb_gap"]    = round(lumo - homo, 6) if (homo and lumo) else None
    desc["xtb_dipole"] = _parse_xtb_dipole(stdout)

    alpha, c6 = _parse_xtb_alpha_c6(stdout)
    desc["xtb_alpha0"] = round(alpha, 4) if alpha is not None else None
    desc["xtb_C6"]     = round(c6,    2) if c6    is not None else None

    # Neutral Mulliken charges (from the neutral SP charges file)
    charges_neutral = _parse_xtb_charges(work_dir / "neutral" / "charges")

    # ── IP (separate call so charges file isn't overwritten) ──────────
    r_ip = _xtb_sp(xyz_str, work_dir / "ip", xtb_bin, charge, ["--vip"])
    if r_ip:
        for line in r_ip.stdout.splitlines():
            if "delta SCC IP (eV)" in line:
                try:
                    desc["xtb_IP"] = float(line.split(":")[-1])
                except ValueError:
                    pass

    # ── EA (separate call) ────────────────────────────────────────────
    r_ea = _xtb_sp(xyz_str, work_dir / "ea", xtb_bin, charge, ["--vea"])
    if r_ea:
        for line in r_ea.stdout.splitlines():
            if "delta SCC EA (eV)" in line:
                try:
                    desc["xtb_EA"] = float(line.split(":")[-1])
                except ValueError:
                    pass

    ip = desc.get("xtb_IP")
    ea = desc.get("xtb_EA")
    if ip is not None and ea is not None:
        mu  = -(ip + ea) / 2
        eta =  (ip - ea) / 2
        desc["xtb_mu"]    = round(mu, 6)
        desc["xtb_eta"]   = round(eta, 6)
        desc["xtb_omega"] = round(mu**2 / (2 * eta), 6) if eta != 0 else None

    # ── Fukui via finite difference (N, N±1 charges) ──────────────────
    fukui = _fukui_finite_diff(xyz_str, symbols, coords, xtb_bin,
                                work_dir / "fukui")

    # ── Assign per-atom values at aldehyde carbon ─────────────────────
    ald_atoms = find_aldehyde_atoms(symbols, coords)
    if ald_atoms:
        c_idx, o_idx, _ = ald_atoms[0]
        q = fukui["q_neutral"] if fukui else charges_neutral
        if q and c_idx < len(q):
            desc["mulliken_CHO_C"] = round(q[c_idx], 6)
        if q and o_idx < len(q):
            desc["mulliken_CHO_O"] = round(q[o_idx], 6)
        desc["wbo_CO"] = _parse_wbo(work_dir / "neutral" / "wbo", c_idx, o_idx)
        if fukui:
            fp = fukui["+"][c_idx]
            fm = fukui["-"][c_idx]
            f0 = fukui["0"][c_idx]
            if fp is not None:
                desc["fukui_plus_CHO_C"]  = fp
                desc["fukui_minus_CHO_C"] = fm
                desc["fukui_0_CHO_C"]     = f0
                desc["dual_descriptor_CHO_C"] = round(fp - fm, 6) if fm is not None else None

    desc["pa_CHO_O"] = _calc_proton_affinity(
        xyz_str, symbols, coords, xtb_bin, work_dir / "pa", desc.get("xtb_energy")
    )

    return desc


# ══════════════════════════════════════════════════════════════════════════
#  Level 2 — Morfeus steric + xTB Python interface
# ══════════════════════════════════════════════════════════════════════════

_MORFEUS_FIELDS = [
    "vbur_CHO_C", "sterimol_L", "sterimol_B1", "sterimol_B5", "SASA_total",
    "P_int",
]


def calc_morfeus(symbols: list[str], coords: np.ndarray) -> dict[str, Any]:
    desc: dict[str, Any] = {f: None for f in _MORFEUS_FIELDS}
    if not HAS_MORFEUS:
        return desc

    ald_atoms = find_aldehyde_atoms(symbols, coords)

    # ── Steric descriptors ─────────────────────────────────────────────
    if ald_atoms:
        c_idx, o_idx, r_idx = ald_atoms[0]
        c_1 = c_idx + 1  # morfeus uses 1-based indexing

        try:
            bv = BuriedVolume(symbols, coords, c_1,
                              include_hs=True, radius=VBUR_RADIUS)
            desc["vbur_CHO_C"] = round(bv.fraction_buried_volume * 100, 4)
        except Exception as e:
            log.debug("BuriedVolume failed: %s", e)

        if r_idx >= 0:
            try:
                sm = Sterimol(symbols, coords, c_1, r_idx + 1)
                desc["sterimol_L"]  = round(sm.L_value, 4)
                desc["sterimol_B1"] = round(sm.B_1_value, 4)
                desc["sterimol_B5"] = round(sm.B_5_value, 4)
            except Exception as e:
                log.debug("Sterimol failed: %s", e)

    # ── SASA ───────────────────────────────────────────────────────────
    try:
        sasa = SASA(symbols, coords)
        desc["SASA_total"] = round(sasa.area, 4)
    except Exception as e:
        log.debug("SASA failed: %s", e)

    # ── Dispersion P_int ───────────────────────────────────────────────
    if HAS_DISP:
        try:
            disp = Dispersion(symbols, coords)
            desc["P_int"] = round(disp.p_int, 4)
        except Exception as e:
            log.debug("Dispersion failed: %s", e)

    return desc


# ══════════════════════════════════════════════════════════════════════════
#  Level 3 — Multiwfn (ESP surface + Hirshfeld Fukui)
# ══════════════════════════════════════════════════════════════════════════

_MWF_FIELDS = [
    # GFN2 wavefunction orbital energies (from molden, should match Level 1 values)
    "wfn_HOMO", "wfn_LUMO", "wfn_gap",
    # ADCH charges (Atomic Dipole Corrected Hirshfeld)
    "adch_CHO_C", "adch_CHO_O",
    # ADCH-based condensed Fukui (more accurate than Mulliken-based Level 1)
    "adch_fukui_plus_CHO_C",   # f⁺ at C: susceptibility to nucleophilic attack
    "adch_fukui_minus_CHO_C",  # f⁻ at C
    "adch_fukui_minus_CHO_O",  # f⁻ at O: proton/electrophile affinity site
    # QTAIM C=O bond critical point (characterises bond polarity and π character)
    "qtaim_rho_CO",   # electron density at BCP (a.u.)
    "qtaim_lap_CO",   # Laplacian ∇²ρ at BCP (a.u.; negative → covalent)
    "qtaim_ell_CO",   # bond ellipticity ε (π character of C=O)
]

# Multiwfn must run from its OWN directory where settings.ini lives.
# Molden file is copied there as a temp file; all paths relative to mwf_install_dir.
#
# Analysis chosen:
#   Option 7 → 1  : Hirshfeld charges  (fast, ~1 s)
#   Option 7 → 11 : ADCH charges       (fast, ~1 s, higher accuracy)
#   ESP surface (option 12) is intentionally omitted from automation —
#   it requires grid integration (~minutes per molecule). Run manually if needed.

_MWF_HIRSHFELD = "7\n1\n1\nn\n0\nq\n"
_MWF_ADCH      = "7\n11\n1\nn\n0\nq\n"
# 2→topology, 3→search BCPs from atomic-pair midpoints, 7→values at CPs, -1→dump to CPprop.txt (no ESP), -10→main, q→quit
_MWF_QTAIM = "2\n3\n7\n-1\n-10\nq\n"


def _run_multiwfn(mwf_bin: str, molden_path, mwf_input: str) -> str:
    """
    Run Multiwfn in the molden's OWN directory (isolated per molecule) so many
    instances can run concurrently without clobbering each other's temp/output
    files. `Multiwfnpath` points at the install dir for settings.ini (Multiwfn
    also runs fine on defaults without it). `molden_path` is the full path.
    """
    molden_path = Path(molden_path)
    env = dict(os.environ)
    env["Multiwfnpath"] = str(Path(mwf_bin).parent)
    try:
        r = subprocess.run(
            [mwf_bin, molden_path.name],
            input=mwf_input, capture_output=True, text=True,
            cwd=str(molden_path.parent), timeout=120, env=env,
        )
        return r.stdout
    except subprocess.TimeoutExpired:
        log.debug("Multiwfn timed out (analysis too slow for this molecule)")
    except Exception as e:
        log.debug("Multiwfn error: %s", e)
    return ""


def _gen_molden(xyz_str: str, work_dir: Path, xtb_bin: str,
                mwf_bin: str, stem: str, charge: int = 0) -> Path | None:
    """
    Generate the xTB molden inside `work_dir` (isolated per molecule) and return
    its full path. Multiwfn is run there directly (see _run_multiwfn) — no copy
    into the shared install dir, so concurrent instances don't collide.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "sp.xyz").write_text(xyz_str, encoding="utf-8")
    r = _run_xtb(
        [xtb_bin, "sp.xyz", "--gfn", "2", "--charge", str(charge),
         "--molden", "--norestart"],
        work_dir, timeout=120,
    )
    src = work_dir / "molden.input"
    if not src.exists():
        return None
    dest = work_dir / f"_tmp_{stem}.molden"
    import shutil as _shutil
    _shutil.copy2(str(src), str(dest))
    return dest



def _parse_esp_output(stdout: str) -> dict[str, Any]:
    # Kept for reference / manual use; not called in the automated pipeline
    desc: dict[str, Any] = {}
    for line in stdout.splitlines():
        if "Global surface minimum" in line:
            try:
                val = float(line.split("=")[-1].split()[0])
                desc["ESP_Vmin_au"]   = round(val, 6)
                desc["ESP_Vmin_kcal"] = round(val * 627.509, 4)
            except (ValueError, IndexError):
                pass
        if "Global surface maximum" in line:
            try:
                val = float(line.split("=")[-1].split()[0])
                desc["ESP_Vmax_au"]   = round(val, 6)
                desc["ESP_Vmax_kcal"] = round(val * 627.509, 4)
            except (ValueError, IndexError):
                pass
    return desc


def _parse_mwf_charges(stdout: str, atom_idx_1based: int) -> float | None:
    """
    Parse atomic charge for a given 1-based atom index from Multiwfn stdout.

    Multiwfn emits a 'Final atomic charges:' block formatted as:
      Atom    1(O ):    -0.26530531
      Atom    2(C ):     0.09947330
    We look for the LAST such block (appears after 'Final atomic charges:')
    because Multiwfn may print intermediate blocks during the iteration.
    """
    import re as _re
    pat = _re.compile(
        r"^\s*Atom\s+" + str(atom_idx_1based) + r"\s*\([^)]+\)\s*:\s*([-\d.]+)"
    )
    in_table = False
    last_val = None
    for line in stdout.splitlines():
        if "Final atomic charges:" in line:
            in_table = True
            continue
        # blank line or next section header ends the table
        if in_table and line.strip() == "":
            in_table = False
        if in_table:
            m = pat.match(line)
            if m:
                try:
                    last_val = round(float(m.group(1)), 6)
                except ValueError:
                    pass
    return last_val


def _parse_molden_homo_lumo(molden_path: Path) -> tuple[float | None, float | None]:
    """
    Parse HOMO and LUMO energies (eV) from an xTB-generated molden file.
    xTB writes orbital energies in Hartree; we multiply by 27.211386 to get eV.
    Returns (homo_eV, lumo_eV).
    """
    if not molden_path.exists():
        return None, None
    HARTREE_TO_EV = 27.211386
    occupied: list[float] = []
    virtual:  list[float] = []
    in_mo = False
    cur_ene: float | None = None
    try:
        for line in molden_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            ll = line.strip().lower()
            if ll == "[mo]":
                in_mo = True
                continue
            if in_mo and ll.startswith("["):
                in_mo = False
                continue
            if not in_mo:
                continue
            if ll.startswith("ene="):
                try:
                    cur_ene = float(line.split("=", 1)[1].strip())
                except (ValueError, IndexError):
                    cur_ene = None
            elif ll.startswith("occup="):
                try:
                    occ = float(line.split("=", 1)[1].strip())
                    if cur_ene is not None:
                        if occ > 0.5:
                            occupied.append(cur_ene)
                        else:
                            virtual.append(cur_ene)
                    cur_ene = None
                except (ValueError, IndexError):
                    pass
    except Exception:
        return None, None
    homo = max(occupied) * HARTREE_TO_EV if occupied else None
    lumo = min(virtual)  * HARTREE_TO_EV if virtual  else None
    return (round(homo, 6) if homo is not None else None,
            round(lumo, 6) if lumo is not None else None)


def _parse_qtaim_bcp_from_file(
    cpfile: Path, c_coord: np.ndarray, o_coord: np.ndarray
) -> dict[str, Any]:
    """
    Parse QTAIM BCP properties from Multiwfn CPprop.txt.
    Locates the C=O BCP as the (3,-1) CP whose position is closest to the C-O midpoint.
    """
    result: dict[str, Any] = {"qtaim_rho_CO": None, "qtaim_lap_CO": None, "qtaim_ell_CO": None}
    if not cpfile.exists():
        return result
    midpoint = (c_coord + o_coord) / 2.0
    try:
        content = cpfile.read_text(encoding="utf-8", errors="ignore")
        # Each CP block header: "-{16+}   CP N,   Type (3,-1)   -{16+}"
        # Capture the body after each (3,-1) header until the next header or EOF
        blocks = re.findall(
            r"-{16,}[^\n]*Type \(3,-1\)[^\n]*-{16,}(.*?)(?=-{16,}|\Z)",
            content, re.DOTALL,
        )
        best_dist, best_props = float("inf"), None
        for block in blocks:
            m = re.search(
                r"Position \(Angstrom\):\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)", block
            )
            if not m:
                continue
            pos = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            dist = float(np.linalg.norm(pos - midpoint))
            if dist >= best_dist:
                continue
            best_dist = dist
            props: dict[str, Any] = {}
            m2 = re.search(r"Density of all electrons:\s*([-\d.E+]+)", block)
            if m2:
                props["qtaim_rho_CO"] = round(float(m2.group(1)), 6)
            m2 = re.search(r"Laplacian of electron density:\s*([-\d.E+]+)", block)
            if m2:
                props["qtaim_lap_CO"] = round(float(m2.group(1)), 6)
            m2 = re.search(r"Ellipticity of electron density:\s*([-\d.E+]+)", block)
            if m2:
                props["qtaim_ell_CO"] = round(float(m2.group(1)), 6)
            best_props = props
        if best_props and best_dist < 0.5:
            result.update(best_props)
    except Exception as e:
        log.debug("CPprop.txt parse error: %s", e)
    return result


def _adch_fukui(
    xyz_str: str, symbols: list[str], coords: np.ndarray,
    xtb_bin: str, mwf_bin: str, work_dir: Path, stem: str,
) -> dict[str, Any]:
    """
    ADCH-based condensed Fukui functions at CHO_C and CHO_O.
    Generates GFN2 moldens for N, N+1 (charge=-1), N-1 (charge=+1) states;
    runs Multiwfn ADCH on each; applies finite-difference formulas.
    """
    result: dict[str, Any] = {
        "adch_fukui_plus_CHO_C":  None,
        "adch_fukui_minus_CHO_C": None,
        "adch_fukui_minus_CHO_O": None,
    }
    ald_atoms = find_aldehyde_atoms(symbols, coords)
    if not ald_atoms:
        return result
    c_idx, o_idx, _ = ald_atoms[0]
    n_atoms = len(symbols)

    def get_all_adch(charge: int, label: str) -> list[float] | None:
        d = work_dir / label
        fname = _gen_molden(xyz_str, d, xtb_bin, mwf_bin,
                            f"{stem}_{label}", charge=charge)
        if fname is None:
            return None
        try:
            out = _run_multiwfn(mwf_bin, fname, _MWF_ADCH)
            charges: list[float] = []
            in_table = False
            for line in out.splitlines():
                if "Final atomic charges:" in line:
                    in_table = True
                    continue
                if in_table and line.strip() == "":
                    in_table = False
                if in_table:
                    m = re.match(r"^\s*Atom\s+\d+\s*\([^)]+\)\s*:\s*([-\d.]+)", line)
                    if m:
                        try:
                            charges.append(float(m.group(1)))
                        except ValueError:
                            pass
            return charges if len(charges) == n_atoms else None
        except Exception as e:
            log.debug("ADCH Fukui %s failed: %s", label, e)
            return None
        finally:
            if fname is not None:
                try:
                    Path(fname).unlink(missing_ok=True)
                except Exception:
                    pass

    q_n   = get_all_adch(0,  "adch_n")
    q_np1 = get_all_adch(-1, "adch_np1")   # N+1 electrons
    q_nm1 = get_all_adch(+1, "adch_nm1")   # N-1 electrons

    if q_n is None:
        return result
    if q_np1 and len(q_np1) == n_atoms:
        result["adch_fukui_plus_CHO_C"]  = round(q_n[c_idx] - q_np1[c_idx], 6)
    if q_nm1 and len(q_nm1) == n_atoms:
        result["adch_fukui_minus_CHO_C"] = round(q_nm1[c_idx] - q_n[c_idx], 6)
        result["adch_fukui_minus_CHO_O"] = round(q_nm1[o_idx] - q_n[o_idx], 6)
    return result


def calc_multiwfn(
    xyz_str: str,
    symbols: list[str],
    coords: np.ndarray,
    xtb_bin: str,
    mwf_bin: str,
    work_dir: Path,
    stem: str = "tmp",
) -> dict[str, Any]:
    """
    Compute wavefunction descriptors via GFN2 molden + Multiwfn:
      - HOMO/LUMO/gap from molden (validates Level 1 values)
      - ADCH charges at CHO_C and CHO_O
      - ADCH-based condensed Fukui (N±1 charge states)
      - QTAIM C=O bond critical point (ρ, ∇²ρ, ε)
    """
    desc: dict[str, Any] = {f: None for f in _MWF_FIELDS}

    # ── Generate GFN2 molden (neutral) ────────────────────────────────────
    molden_fname = _gen_molden(xyz_str, work_dir / "wfn_n", xtb_bin, mwf_bin, stem)
    if molden_fname is None:
        log.debug("No molden generated; skipping Multiwfn")
        return desc

    # ── HOMO/LUMO from molden (direct parse, no Multiwfn needed) ─────────
    local_molden = work_dir / "wfn_n" / "molden.input"
    homo_wfn, lumo_wfn = _parse_molden_homo_lumo(local_molden)
    desc["wfn_HOMO"] = homo_wfn
    desc["wfn_LUMO"] = lumo_wfn
    desc["wfn_gap"]  = round(lumo_wfn - homo_wfn, 6) if (homo_wfn and lumo_wfn) else None

    ald_atoms = find_aldehyde_atoms(symbols, coords)
    if not ald_atoms:
        log.debug("No aldehyde atoms found; skipping Multiwfn per-atom analyses")
    else:
        c_1based = ald_atoms[0][0] + 1
        o_1based = ald_atoms[0][1] + 1

        # ── ADCH charges ─────────────────────────────────────────────────
        try:
            out_a = _run_multiwfn(mwf_bin, molden_fname, _MWF_ADCH)
            desc["adch_CHO_C"] = _parse_mwf_charges(out_a, c_1based)
            desc["adch_CHO_O"] = _parse_mwf_charges(out_a, o_1based)
        except Exception as e:
            log.debug("Multiwfn ADCH failed: %s", e)

        # ── QTAIM topology ────────────────────────────────────────────────
        try:
            # Multiwfn writes CPprop.txt to its cwd, which is now the molden's
            # own (per-molecule) directory — isolated, so concurrent runs don't
            # overwrite each other's CPprop.txt.
            cpfile = molden_fname.parent / "CPprop.txt"
            cpfile.unlink(missing_ok=True)
            _run_multiwfn(mwf_bin, molden_fname, _MWF_QTAIM)
            c_coord = coords[ald_atoms[0][0]]
            o_coord = coords[ald_atoms[0][1]]
            qtaim = _parse_qtaim_bcp_from_file(cpfile, c_coord, o_coord)
            desc.update(qtaim)
            cpfile.unlink(missing_ok=True)
        except Exception as e:
            log.debug("Multiwfn QTAIM failed: %s", e)

        # Clean up the neutral molden (lives in its per-molecule work dir).
        try:
            Path(molden_fname).unlink(missing_ok=True)
        except Exception:
            pass

        # ── ADCH Fukui (N±1 charge states, separate moldens) ─────────────
        try:
            fukui = _adch_fukui(
                xyz_str, symbols, coords, xtb_bin, mwf_bin,
                work_dir / "adch_fukui", stem,
            )
            desc.update(fukui)
        except Exception as e:
            log.debug("ADCH Fukui failed: %s", e)

    return desc


# ══════════════════════════════════════════════════════════════════════════
#  Full pipeline per molecule
# ══════════════════════════════════════════════════════════════════════════

def _auto_nconfs(rotbonds: int, n_confs_max: int) -> int:
    """Scale conformer count linearly with RotBonds: 1 at rb≤3, n_confs_max at rb≥9."""
    if rotbonds <= 3:
        return 1
    frac = min(1.0, (rotbonds - 3) / 6)
    return min(n_confs_max, max(3, round(n_confs_max * frac)))


def _stem(name: str, idx: int) -> str:
    return f"{idx:06d}_{re.sub(r'[^\w-]', '_', name)[:50]}"


def process_one(
    rec: dict,
    idx: int,
    xyz_dir: Path,
    work_dir: Path,
    xtb_bin: str | None,
    mwf_bin: str | None,
    do_xtb_opt: bool,
    do_multiwfn: bool,
    n_confs: int = 1,
) -> dict[str, Any]:
    index_val   = str(rec.get("index") or idx).strip()
    smiles      = (rec.get("SMILES") or rec.get("smiles") or "").strip()
    pubchem_cid = str(rec.get("PubChem_CID") or "").strip()

    base: dict[str, Any] = {
        "index": index_val, "SMILES": smiles, "PubChem_CID": pubchem_cid,
        "xtb_optimized": False, "error": "",
    }

    # ── Level 0: RDKit ─────────────────────────────────────────────────
    base.update(calc_rdkit(smiles))

    # ── 3D geometry ────────────────────────────────────────────────────
    stem_str   = _stem(index_val, idx)
    xtb_dir    = work_dir / stem_str

    # Multi-conformer search only when xTB opt is available AND molecule is flexible
    rotbonds   = base.get("RotBonds") or 0
    use_nconfs = _auto_nconfs(rotbonds, n_confs) if (n_confs > 1 and xtb_bin and do_xtb_opt) else 1

    if use_nconfs == 1:
        mol = smiles_to_3d(smiles)
        if mol is None:
            base["error"] = "3D_embed_failed"
            return base
        xyz_str = mol_to_xyz_str(mol, index_val)
    else:
        xyz_list = smiles_to_3d_multi(smiles, use_nconfs, index_val)
        if not xyz_list:
            base["error"] = "3D_embed_failed"
            return base
        log.info("[%d] idx=%s: %d conformers (RotBonds=%d)",
                 idx, index_val, len(xyz_list), rotbonds)

    # ── Level 1: xTB opt + descriptors ────────────────────────────────
    if xtb_bin:
        if do_xtb_opt:
            if use_nconfs == 1:
                opt = xtb_optimize(xyz_str, xtb_dir / "opt", xtb_bin)
                if opt:
                    xyz_str = opt
                    base["xtb_optimized"] = True
                else:
                    log.debug("[%d] xTB opt failed, using MMFF geometry", idx)
            else:
                best, optimized = pick_best_xtb_conformer(
                    xyz_list, xtb_dir / "confs", xtb_bin)
                xyz_str = best
                base["xtb_optimized"] = optimized

        symbols, coords = parse_xyz(xyz_str)
        base.update(calc_xtb(xyz_str, symbols, coords, xtb_bin,
                              xtb_dir / "sp"))
    else:
        symbols, coords = parse_xyz(xyz_str)

    # Save final XYZ
    xyz_out = xyz_dir / f"{stem_str}.xyz"
    xyz_out.write_text(xyz_str, encoding="utf-8")
    base["xyz_file"] = str(xyz_out)

    # ── Level 2: Morfeus ───────────────────────────────────────────────
    base.update(calc_morfeus(symbols, coords))

    # ── Level 3: Multiwfn ─────────────────────────────────────────────
    if do_multiwfn and xtb_bin and mwf_bin:
        base.update(calc_multiwfn(
            xyz_str, symbols, coords, xtb_bin, mwf_bin,
            work_dir / stem_str / "mwf",
            stem=stem_str,
        ))

    return base


# ══════════════════════════════════════════════════════════════════════════
#  CLI / main
# ══════════════════════════════════════════════════════════════════════════

_ALL_FIELDS = (
    ["index", "SMILES", "PubChem_CID", "xtb_optimized", "error", "xyz_file"] +
    _RDKIT_FIELDS + _XTB_FIELDS + _MORFEUS_FIELDS + _MWF_FIELDS
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute multi-level descriptors for aldehydes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",   default="aldehydes.csv")
    ap.add_argument("--smiles",  action="append", metavar="SMILES")
    ap.add_argument("--output",  default="ald_desc/descriptors.csv")
    ap.add_argument("--work-dir",default="ald_desc", help="Working directory")
    ap.add_argument("--max",     type=int, default=0)
    ap.add_argument("--xtb-bin", default=shutil.which("xtb"),
                    help="Path to xtb executable")
    ap.add_argument("--multiwfn-bin",
                    default=shutil.which("Multiwfn") or shutil.which("multiwfn"),
                    help="Path to Multiwfn executable")
    ap.add_argument("--no-xtb-opt",  action="store_true",
                    help="Skip xTB geometry optimization (use MMFF only)")
    ap.add_argument("--multiwfn",    action="store_true",
                    help="Enable Multiwfn ESP/Fukui analysis (requires --multiwfn-bin)")
    ap.add_argument("--n-confs",     type=int, default=1, metavar="N",
                    help="ETKDG conformers to generate for flexible molecules "
                         "(RotBonds > 3); best xTB energy is kept. Default: 1")
    args = ap.parse_args()

    log.info("── Dependency check ──────────────────────────────────")
    log.info("  RDKit    : OK")
    log.info("  morfeus  : %s", "OK" if HAS_MORFEUS else "NOT FOUND  →  pip install morfeus-ml")
    log.info("  xtb CLI  : %s", args.xtb_bin or "NOT FOUND  →  conda install -c conda-forge xtb")
    log.info("  Multiwfn : %s", args.multiwfn_bin or "NOT FOUND  →  http://sobereva.com/multiwfn/")
    log.info("─────────────────────────────────────────────────────")

    work_dir = Path(args.work_dir)
    xyz_dir  = work_dir / "xyz"
    for d in (work_dir, xyz_dir):
        d.mkdir(exist_ok=True)

    if args.smiles:
        records = [{"index": str(i), "SMILES": s, "PubChem_CID": ""}
                   for i, s in enumerate(args.smiles)]
    else:
        csv_path = Path(args.input)
        if not csv_path.exists():
            log.error("Input not found: %s  (use --smiles for a quick test)", csv_path)
            return 1
        with csv_path.open(encoding="utf-8") as fh:
            records = list(csv.DictReader(fh))
        log.info("Loaded %d records from %s", len(records), csv_path)

    if args.max:
        records = records[: args.max]
        log.info("Capped at %d records", len(records))

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)

    n_ok = n_fail = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ALL_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for idx, rec in enumerate(tqdm(records, desc="Computing descriptors")):
            try:
                row = process_one(
                    rec, idx, xyz_dir, work_dir,
                    xtb_bin=args.xtb_bin,
                    mwf_bin=args.multiwfn_bin,
                    do_xtb_opt=not args.no_xtb_opt,
                    do_multiwfn=args.multiwfn and bool(args.multiwfn_bin),
                    n_confs=args.n_confs,
                )
            except Exception as e:
                log.warning("[%d] Unexpected error: %s", idx, e)
                row = {"index": rec.get("index", ""), "SMILES": rec.get("SMILES", ""),
                       "PubChem_CID": rec.get("PubChem_CID", ""), "error": str(e)}
            writer.writerow(row)
            fh.flush()
            n_ok += 1 if not row.get("error") else 0
            n_fail += 1 if row.get("error") else 0

    log.info("─────────────────────────────────────────────────────")
    log.info("Processed : %d OK, %d failed", n_ok, n_fail)
    log.info("Output    : %s", out_path.resolve())
    log.info("XYZ dir   : %s", xyz_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
