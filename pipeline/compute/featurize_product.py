#!/usr/bin/env python3
"""
Cross/Homo-Benzoin PRODUCT featurizer — QM descriptors + xTB ΔG (no DFT/ORCA).
==============================================================================
Sibling of featurize_screen.py, but for the benzoin *product* of an aldehyde PAIR
instead of a single aldehyde. One row per ordered pair (donor, acceptor):

    D-CHO + A-CHO  ->  D-C(=O)-CH(OH)-A
       donor   acceptor    (α-hydroxy ketone)

The donor carbonyl C becomes the KETONE carbon; the acceptor carbonyl C becomes
the CARBINOL carbon. Homo-benzoin is just donor == acceptor (one product).

Descriptors mirror the aldehyde tiers (ald_descriptors_qm) and reuse its low-level
xTB / morfeus / Multiwfn machinery, but are re-anchored from the single CHO carbon
to the product's TWO carbonyl-derived sites plus the intramolecular H-bond:

    sites : ketC, ketO, carbC, hydO, hydH
    bonds : CO_ket (ketC=ketO), CC_new (ketC-carbC), CO_carb (carbC-hydO)
    HB    : hydH···ketO  (universal α-hydroxy-ketone 5-membered H-bond)

  xTB     : global (HOMO/LUMO/gap/IP/EA/mu/eta/omega/dipole), Mulliken charges +
            Mulliken-Fukui at ketC/carbC, WBO of the 3 bonds, PA at ketO
  morfeus : %Vbur at ketC/carbC, Sterimol on CC_new, SASA, dispersion P_int,
            H-bond geometry (distance/angle) + core dihedral
  Multiwfn: ADCH charges, ADCH-Fukui at ketC/carbC, QTAIM BCP (rho/lap/ell) at
            CO_ket / CC_new and rho at the H-bond BCP   [--multiwfn, slow]
  ΔG      : xTB --ohess (ALPB) free energies; dG = G(prod) - G(donor) - G(acceptor)
            per-aldehyde G is cached so it is computed once per unique aldehyde.

Usage
-----
  # quick test: a few explicit pairs "donor>>acceptor"
  python featurize_product.py \
      --pair "O=Cc1ccccc1>>O=Cc1ccc(cc1)[N+](=O)[O-]" \
      --pair "O=Cc1ccco1>>O=Cc1ccccc1" \
      --output /tmp/prod.csv --xtb-bin /home/schen3/xtb/bin/xtb --n-confs 3

  # a CSV of pairs (cols: donor_smiles, acceptor_smiles[, donor_id, acceptor_id])
  python featurize_product.py --input pairs.csv --output out.csv \
      --xtb-bin xtb --solvent dmso --n-confs 10 --workers 12 --xtb-cores 2
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# conf_funnel_v3 BEFORE thermo_orca/ald_descriptors_qm: the funnel↔thermo_orca imports
# are mutually circular (thermo_orca imports conf_funnel_v3 at top level; conf_funnel_v2
# imports thermo_orca symbols). Loading conf_funnel_v3 first fully initialises thermo_orca
# via its chain; the reverse order ImportErrors on _mol_rotbonds when run as __main__.
import conf_funnel_v3
import ald_descriptors_qm as A
import thermo_orca as Th


def _ranker(name: str):
    """Conformer-ranking function: funnel_v3 (deterministic + RMSD + topology guard,
    the production fix for ΔG conformer noise) or the plain xTB ranker."""
    if name == "funnel_v3":
        return conf_funnel_v3.rank_conformers_funnel_v3
    return Th._rank_conformers

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

log = logging.getLogger("featurize_product")

HARTREE_TO_KCAL = 627.509474
VBUR_RADIUS = 3.5

# donor (reactant 1) -> ketone ; acceptor (reactant 2) -> carbinol
PROD_RXN = AllChem.ReactionFromSmarts(
    "[CX3H1:1](=[O:2]).[CX3H1:3](=[O:4])>>[C:1](=[O:2])[CH1:3]([OX2H1:4])"
)

# ── Output schema ───────────────────────────────────────────────────────────
_META = ["index", "donor_smiles", "acceptor_smiles", "product_smiles",
         "donor_class", "acceptor_class", "reaction_type", "is_homo",
         "xtb_optimized", "error", "xyz_file"]

_XTB = [
    "xtb_energy", "xtb_HOMO", "xtb_LUMO", "xtb_gap",
    "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta", "xtb_omega", "xtb_dipole",
    "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO", "mulliken_hydH",
    "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb",
    "fukui_plus_ketC", "fukui_minus_ketC", "fukui_0_ketC", "dual_ketC",
    "fukui_plus_carbC", "fukui_minus_carbC", "fukui_0_carbC", "dual_carbC",
    "pa_ketO",
]

_MORF = [
    "vbur_ketC", "vbur_carbC", "sterimol_L", "sterimol_B1", "sterimol_B5",
    "SASA_total", "P_int", "hb_dist", "hb_angle", "dih_core",
]

_MWF = [
    "adch_ketC", "adch_ketO", "adch_carbC", "adch_hydO", "adch_hydH",
    "adch_fukui_plus_ketC", "adch_fukui_minus_ketC",
    "adch_fukui_plus_carbC", "adch_fukui_minus_carbC",
    "qtaim_rho_CO_ket", "qtaim_lap_CO_ket", "qtaim_ell_CO_ket",
    "qtaim_rho_CC_new", "qtaim_lap_CC_new", "qtaim_ell_CC_new",
    "qtaim_rho_HB",
]

_DG = ["G_donor", "G_acceptor", "G_product", "dG_xtb_kcal"]

OUT_FIELDS = _META + _XTB + _MORF + _MWF + _DG

# Aldehyde-side schema (same descriptors as the screen, but on the funnel_v3
# geometry, so aldehyde and product descriptors are method-consistent).
ALD_FIELDS = list(A._ALL_FIELDS) + ["G_ald_xtb"]


# ══════════════════════════════════════════════════════════════════════════
#  Product construction + classification
# ══════════════════════════════════════════════════════════════════════════

def build_product(donor_smi: str, acceptor_smi: str) -> str | None:
    d = Chem.MolFromSmiles(donor_smi)
    a = Chem.MolFromSmiles(acceptor_smi)
    if d is None or a is None:
        return None
    try:
        prods = PROD_RXN.RunReactants((d, a))
    except Exception:
        return None
    for t in prods:
        p = t[0]
        try:
            Chem.SanitizeMol(p)
            return Chem.MolToSmiles(p, isomericSmiles=False)
        except Exception:
            continue
    return None


def classify(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return "unknown"
    arom_c = any(a.GetIsAromatic() for a in m.GetAtoms())
    if not arom_c:
        return "aliphatic"
    hetero = any(a.GetIsAromatic() and a.GetSymbol() != "C" for a in m.GetAtoms())
    return "aromatic_hetero" if hetero else "aromatic_carbo"


def reaction_type(dc: str, ac: str, is_homo: bool) -> str:
    if is_homo:
        return "homo"
    short = {"aliphatic": "aliph", "aromatic_carbo": "carbo",
             "aromatic_hetero": "hetero", "unknown": "unk"}
    return "-".join(sorted((short.get(dc, "unk"), short.get(ac, "unk"))))


# ══════════════════════════════════════════════════════════════════════════
#  Geometric benzoin-core finder  (O=C-C(-OH)H motif)
# ══════════════════════════════════════════════════════════════════════════

def find_benzoin_core(symbols: list[str], coords: np.ndarray) -> dict | None:
    """Locate ketC/ketO/carbC/hydO/hydH of the α-hydroxy ketone formed by benzoin.
    Returns 0-based indices, or None if the motif is not found."""
    n = len(symbols)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))

    def has_h(o: int) -> bool:
        return any(symbols[h] == "H" and dist[o, h] < A.CH_BOND_MAX for h in range(n))

    for c in range(n):
        if symbols[c] != "C":
            continue
        # ketone oxygen: C=O (short) with NO hydrogen on O
        keto = [o for o in range(n) if symbols[o] == "O"
                and dist[c, o] < A.CO_BOND_MAX and not has_h(o)]
        if len(keto) != 1:
            continue
        keto = keto[0]
        # carbinol carbon: a C neighbor bearing a hydroxyl (C-OH) and at least one H
        for cb in range(n):
            if cb == c or symbols[cb] != "C" or dist[c, cb] >= 1.75:
                continue
            hyd = [o for o in range(n) if symbols[o] == "O"
                   and 1.20 < dist[cb, o] < 1.65 and has_h(o)]
            if not hyd:
                continue
            hydO = hyd[0]
            hydH = next((h for h in range(n) if symbols[h] == "H"
                         and dist[hydO, h] < A.CH_BOND_MAX), None)
            if hydH is None:
                continue
            return {"ketC": c, "ketO": keto, "carbC": cb, "hydO": hydO, "hydH": hydH}
    return None


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c in degrees."""
    v1, v2 = a - b, c - b
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


def _dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


# ══════════════════════════════════════════════════════════════════════════
#  xTB tier (global + per-site at ketC/carbC)
# ══════════════════════════════════════════════════════════════════════════

def calc_xtb_product(xyz_str, symbols, coords, core, xtb_bin, work_dir) -> dict:
    d: dict[str, Any] = {f: None for f in _XTB}

    r_sp = A._xtb_sp(xyz_str, work_dir / "neutral", xtb_bin, 0)
    if r_sp is None:
        return d
    out = r_sp.stdout
    d["xtb_energy"] = A._parse_xtb_energy(out)
    homo, lumo = A._parse_xtb_homo_lumo(out)
    d["xtb_HOMO"], d["xtb_LUMO"] = homo, lumo
    d["xtb_gap"] = round(lumo - homo, 6) if (homo and lumo) else None
    d["xtb_dipole"] = A._parse_xtb_dipole(out)
    q_neutral = A._parse_xtb_charges(work_dir / "neutral" / "charges")

    r_ip = A._xtb_sp(xyz_str, work_dir / "ip", xtb_bin, 0, ["--vip"])
    if r_ip:
        for line in r_ip.stdout.splitlines():
            if "delta SCC IP (eV)" in line:
                try: d["xtb_IP"] = float(line.split(":")[-1])
                except ValueError: pass
    r_ea = A._xtb_sp(xyz_str, work_dir / "ea", xtb_bin, 0, ["--vea"])
    if r_ea:
        for line in r_ea.stdout.splitlines():
            if "delta SCC EA (eV)" in line:
                try: d["xtb_EA"] = float(line.split(":")[-1])
                except ValueError: pass
    ip, ea = d.get("xtb_IP"), d.get("xtb_EA")
    if ip is not None and ea is not None:
        mu, eta = -(ip + ea) / 2, (ip - ea) / 2
        d["xtb_mu"], d["xtb_eta"] = round(mu, 6), round(eta, 6)
        d["xtb_omega"] = round(mu ** 2 / (2 * eta), 6) if eta != 0 else None

    fukui = A._fukui_finite_diff(xyz_str, symbols, coords, xtb_bin, work_dir / "fukui")
    q = fukui["q_neutral"] if fukui else q_neutral

    ketC, ketO, carbC = core["ketC"], core["ketO"], core["carbC"]
    hydO, hydH = core["hydO"], core["hydH"]
    if q:
        for name, i in [("ketC", ketC), ("ketO", ketO), ("carbC", carbC),
                        ("hydO", hydO), ("hydH", hydH)]:
            if i < len(q):
                d[f"mulliken_{name}"] = round(q[i], 6)
    wbo = work_dir / "neutral" / "wbo"
    d["wbo_CO_ket"] = A._parse_wbo(wbo, ketC, ketO)
    d["wbo_CC_new"] = A._parse_wbo(wbo, ketC, carbC)
    d["wbo_CO_carb"] = A._parse_wbo(wbo, carbC, hydO)
    if fukui:
        for name, i in [("ketC", ketC), ("carbC", carbC)]:
            fp, fm, f0 = fukui["+"][i], fukui["-"][i], fukui["0"][i]
            if fp is not None:
                d[f"fukui_plus_{name}"] = fp
                d[f"fukui_minus_{name}"] = fm
                d[f"fukui_0_{name}"] = f0
                d[f"dual_{name}"] = round(fp - fm, 6) if fm is not None else None

    d["pa_ketO"] = _proton_affinity(xyz_str, symbols, coords, ketC, ketO,
                                    xtb_bin, work_dir / "pa", d.get("xtb_energy"))
    return d


def _proton_affinity(xyz_str, symbols, coords, c_idx, o_idx, xtb_bin, work_dir, e_neutral):
    if e_neutral is None:
        return None
    o_pos, c_pos = coords[o_idx], coords[c_idx]
    unit = (o_pos - c_pos) / (np.linalg.norm(o_pos - c_pos) + 1e-12)
    h_pos = o_pos + unit * 0.97
    syms = list(symbols) + ["H"]
    xyz = np.vstack([coords, h_pos])
    lines = [str(len(syms)), "protonated_ketO"]
    for s, p in zip(syms, xyz):
        lines.append(f"{s}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}")
    _, e_prot = A._xtb_opt_energy("\n".join(lines), work_dir, xtb_bin, charge=1)
    if e_prot is None:
        return None
    return round((e_neutral - e_prot) * HARTREE_TO_KCAL, 4)


# ══════════════════════════════════════════════════════════════════════════
#  morfeus tier (+ H-bond geometry)
# ══════════════════════════════════════════════════════════════════════════

def calc_morfeus_product(symbols, coords, core) -> dict:
    d: dict[str, Any] = {f: None for f in _MORF}
    if HAS_MORFEUS:
        ketC1, carbC1 = core["ketC"] + 1, core["carbC"] + 1
        try:
            d["vbur_ketC"] = round(BuriedVolume(symbols, coords, ketC1,
                                   include_hs=True, radius=VBUR_RADIUS).fraction_buried_volume * 100, 4)
        except Exception as e:
            log.debug("vbur ketC: %s", e)
        try:
            d["vbur_carbC"] = round(BuriedVolume(symbols, coords, carbC1,
                                    include_hs=True, radius=VBUR_RADIUS).fraction_buried_volume * 100, 4)
        except Exception as e:
            log.debug("vbur carbC: %s", e)
        try:
            sm = Sterimol(symbols, coords, ketC1, carbC1)   # CC_new axis
            d["sterimol_L"] = round(sm.L_value, 4)
            d["sterimol_B1"] = round(sm.B_1_value, 4)
            d["sterimol_B5"] = round(sm.B_5_value, 4)
        except Exception as e:
            log.debug("Sterimol: %s", e)
        try:
            d["SASA_total"] = round(SASA(symbols, coords).area, 4)
        except Exception as e:
            log.debug("SASA: %s", e)
        if HAS_DISP:
            try:
                d["P_int"] = round(Dispersion(symbols, coords).p_int, 4)
            except Exception as e:
                log.debug("Dispersion: %s", e)

    # H-bond geometry hydH···ketO  (donor hydO-hydH ... acceptor ketO)
    hydH, ketO, hydO = coords[core["hydH"]], coords[core["ketO"]], coords[core["hydO"]]
    d["hb_dist"] = round(float(np.linalg.norm(hydH - ketO)), 4)
    d["hb_angle"] = round(_angle(hydO, hydH, ketO), 4)
    d["dih_core"] = round(_dihedral(coords[core["ketO"]], coords[core["ketC"]],
                                    coords[core["carbC"]], coords[core["hydO"]]), 4)
    return d


# ══════════════════════════════════════════════════════════════════════════
#  Multiwfn tier (ADCH charges + Fukui, QTAIM BCPs)   [optional, slow]
# ══════════════════════════════════════════════════════════════════════════

def _bcp_at(cpfile: Path, midpoint: np.ndarray, tol: float = 0.5,
            exclude: list | None = None, max_rho: float | None = None) -> dict:
    """Nearest (3,-1) BCP to `midpoint` (within `tol` Å); return rho/lap/ell/pos.
    `exclude`: skip CPs within 0.3 Å of any of these positions (e.g. covalent BCPs).
    `max_rho`: skip CPs with rho above this (an H-bond BCP is weak, ρ≲0.06)."""
    import re
    res = {"rho": None, "lap": None, "ell": None, "pos": None}
    if not cpfile.exists():
        return res
    content = cpfile.read_text(encoding="utf-8", errors="ignore")
    blocks = re.findall(r"-{16,}[^\n]*Type \(3,-1\)[^\n]*-{16,}(.*?)(?=-{16,}|\Z)",
                        content, re.DOTALL)
    best_d, best, best_pos = float("inf"), None, None
    for b in blocks:
        m = re.search(r"Position \(Angstrom\):\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)", b)
        if not m:
            continue
        pos = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        if exclude and any(e is not None and np.linalg.norm(pos - e) < 0.3 for e in exclude):
            continue
        dd = float(np.linalg.norm(pos - midpoint))
        if dd >= best_d:
            continue
        vals = {}
        for key, pat in [("rho", r"Density of all electrons:\s*([-\d.E+]+)"),
                         ("lap", r"Laplacian of electron density:\s*([-\d.E+]+)"),
                         ("ell", r"Ellipticity of electron density:\s*([-\d.E+]+)")]:
            mm = re.search(pat, b)
            if mm:
                vals[key] = round(float(mm.group(1)), 6)
        if max_rho is not None and vals.get("rho") is not None and vals["rho"] > max_rho:
            continue
        best_d, best, best_pos = dd, vals, pos
    if best and best_d < tol:
        res.update(best); res["pos"] = best_pos
    return res


def calc_multiwfn_product(xyz_str, symbols, coords, core, xtb_bin, mwf_bin, work_dir, stem) -> dict:
    d: dict[str, Any] = {f: None for f in _MWF}
    molden = A._gen_molden(xyz_str, work_dir / "wfn_n", xtb_bin, mwf_bin, stem)
    if molden is None:
        return d
    idx1 = {k: core[k] + 1 for k in core}
    try:
        out_a = A._run_multiwfn(mwf_bin, molden, A._MWF_ADCH)
        for name in ("ketC", "ketO", "carbC", "hydO", "hydH"):
            d[f"adch_{name}"] = A._parse_mwf_charges(out_a, idx1[name])
    except Exception as e:
        log.debug("ADCH charges: %s", e)
    try:
        cpfile = molden.parent / "CPprop.txt"
        cpfile.unlink(missing_ok=True)
        A._run_multiwfn(mwf_bin, molden, A._MWF_QTAIM)
        cov_pos = []
        for tag, (i, j) in {"CO_ket": (core["ketC"], core["ketO"]),
                            "CC_new": (core["ketC"], core["carbC"])}.items():
            bcp = _bcp_at(cpfile, (coords[i] + coords[j]) / 2)
            d[f"qtaim_rho_{tag}"] = bcp["rho"]
            d[f"qtaim_lap_{tag}"] = bcp["lap"]
            d[f"qtaim_ell_{tag}"] = bcp["ell"]
            cov_pos.append(bcp["pos"])
        # H-bond BCP only if the geometry actually has an H-bond (hydH···ketO short);
        # exclude covalent BCPs and cap rho so it can't grab the C-C/C=O bond.
        hb_dist = float(np.linalg.norm(coords[core["hydH"]] - coords[core["ketO"]]))
        if hb_dist < 2.6:
            hb = _bcp_at(cpfile, (coords[core["hydH"]] + coords[core["ketO"]]) / 2,
                         tol=0.6, exclude=cov_pos, max_rho=0.12)
            d["qtaim_rho_HB"] = hb["rho"]
        cpfile.unlink(missing_ok=True)
    except Exception as e:
        log.debug("QTAIM: %s", e)
    try:
        Path(molden).unlink(missing_ok=True)
    except Exception:
        pass
    # ADCH Fukui at ketC/carbC
    try:
        fk = _adch_fukui_product(xyz_str, symbols, coords, core, xtb_bin, mwf_bin,
                                 work_dir / "adch_fukui", stem)
        d.update(fk)
    except Exception as e:
        log.debug("ADCH Fukui: %s", e)
    return d


def _adch_fukui_product(xyz_str, symbols, coords, core, xtb_bin, mwf_bin, work_dir, stem) -> dict:
    import re
    res = {"adch_fukui_plus_ketC": None, "adch_fukui_minus_ketC": None,
           "adch_fukui_plus_carbC": None, "adch_fukui_minus_carbC": None}
    n = len(symbols)

    def adch_all(charge, label):
        fn = A._gen_molden(xyz_str, work_dir / label, xtb_bin, mwf_bin,
                           f"{stem}_{label}", charge=charge)
        if fn is None:
            return None
        try:
            out = A._run_multiwfn(mwf_bin, fn, A._MWF_ADCH)
            ch, intab = [], False
            for line in out.splitlines():
                if "Final atomic charges:" in line:
                    intab = True; continue
                if intab and line.strip() == "":
                    intab = False
                if intab:
                    m = re.match(r"^\s*Atom\s+\d+\s*\([^)]+\)\s*:\s*([-\d.]+)", line)
                    if m:
                        ch.append(float(m.group(1)))
            return ch if len(ch) == n else None
        finally:
            Path(fn).unlink(missing_ok=True)

    q_n, q_np1, q_nm1 = adch_all(0, "n"), adch_all(-1, "np1"), adch_all(1, "nm1")
    if q_n is None:
        return res
    for name, i in [("ketC", core["ketC"]), ("carbC", core["carbC"])]:
        if q_np1:
            res[f"adch_fukui_plus_{name}"] = round(q_n[i] - q_np1[i], 6)
        if q_nm1:
            res[f"adch_fukui_minus_{name}"] = round(q_nm1[i] - q_n[i], 6)
    return res


# ══════════════════════════════════════════════════════════════════════════
#  Per-aldehyde free energy (cached) + ΔG
# ══════════════════════════════════════════════════════════════════════════

def ald_free_energy(smi, work_dir, xtb_bin, solvent, n_confs, T, P, cores, jobs, timeout,
                    conformer="funnel_v3"):
    """Best-conformer xTB --ohess G (Eh) of an aldehyde, and its geometry.
    Returns (G_Eh | None, xyz_block | None).

    Deletes its own conformer-search scratch (work_dir) before returning: G and the
    xyz block are both materialised in memory, so nothing under work_dir is needed
    afterward. Without this, all ~50 unique-aldehyde scratch trees (each hundreds of
    tiny xTB conformer dirs) accumulate for the whole task and, with up to 8 array
    tasks sharing a genoa node's /scratch-local, blow its per-user inode quota."""
    try:
        ranked = _ranker(conformer)(smi, work_dir / "conf", xtb_bin, n_confs, "ald",
                                    solvent=solvent, cores=cores, workers=jobs)
        if not ranked:
            return None, None
        sa, _ = Th.run_ohess(ranked[0][0], work_dir / "ohess", xtb_bin, T, P,
                             solvent=solvent, cores=cores, timeout=timeout)
        return Th.parse_xtb_G(sa), ranked[0][0]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def featurize_aldehyde(smi, idx, *, work_dir, ald_xyz_dir, xtb_bin, mwf_bin, do_multiwfn,
                       solvent, n_confs, T, P, cores, jobs, timeout, conformer="funnel_v3"):
    """Aldehyde geometry (funnel_v3, SAVED) + descriptors (xtb/morfeus/Multiwfn) + G.
    Same method + descriptor schema as the product side, so the two are consistent.
    Returns (row, G_Eh)."""
    row = {f: None for f in ALD_FIELDS}
    row.update({"index": str(idx), "SMILES": smi, "xtb_optimized": False, "error": ""})
    ranked = _ranker(conformer)(smi, work_dir / "conf", xtb_bin, n_confs, "ald",
                                solvent=solvent, cores=cores, workers=jobs)
    if not ranked:
        row["error"] = "ald_embed_failed"
        return row, None
    best = ranked[0][0]
    row["xtb_optimized"] = True
    xyz_out = ald_xyz_dir / f"a{idx:06d}.xyz"
    xyz_out.write_text(best, encoding="utf-8")
    row["xyz_file"] = str(xyz_out)
    symbols, coords = A.parse_xyz(best)
    row.update(A.calc_xtb(best, symbols, coords, xtb_bin, work_dir / "desc_xtb"))
    row.update(A.calc_morfeus(symbols, coords))
    if do_multiwfn and mwf_bin:
        row.update(A.calc_multiwfn(best, symbols, coords, xtb_bin, mwf_bin,
                                   work_dir / "mwf", stem=f"a{idx:06d}"))
    sa, _ = Th.run_ohess(best, work_dir / "ohess", xtb_bin, T, P,
                         solvent=solvent, cores=cores, timeout=timeout)
    G = Th.parse_xtb_G(sa)
    row["G_ald_xtb"] = G
    return row, G


def featurize_pair(rec, idx, *, g_cache, work_dir, xyz_dir, xtb_bin, mwf_bin,
                   do_multiwfn, solvent, n_confs, T, P, cores, jobs, timeout,
                   conformer="funnel_v3"):
    donor = (rec.get("donor_smiles") or "").strip()
    acceptor = (rec.get("acceptor_smiles") or "").strip()
    row = {f: None for f in OUT_FIELDS}
    row.update({"index": str(rec.get("index") or idx), "donor_smiles": donor,
                "acceptor_smiles": acceptor, "xtb_optimized": False, "error": ""})
    if not donor or not acceptor:
        row["error"] = "missing_smiles"; return row

    is_homo = Chem.CanonSmiles(donor) == Chem.CanonSmiles(acceptor)
    dc, ac = classify(donor), classify(acceptor)
    row.update({"donor_class": dc, "acceptor_class": ac, "is_homo": is_homo,
                "reaction_type": reaction_type(dc, ac, is_homo)})

    prod = build_product(donor, acceptor)
    if not prod:
        row["error"] = "product_build_failed"; return row
    row["product_smiles"] = prod

    stem = f"p{idx:06d}"
    mdir = work_dir / stem
    # All descriptors + dG are extracted into `row` (and the product xyz is written to the
    # separate xyz_dir), so mdir's conformer/xTB scratch is dead weight once we return.
    # rmtree it in finally — otherwise every product's scratch tree (hundreds of tiny xTB
    # dirs) lives for the whole task and, with up to 8 array tasks per genoa node, exhausts
    # /scratch-local's per-user inode quota (Errno 122 mid-run).
    try:
        ranked = _ranker(conformer)(prod, mdir / "prod_conf", xtb_bin, n_confs, "prod",
                                    solvent=solvent, cores=cores, workers=jobs)
        if not ranked:
            row["error"] = "prod_embed_failed"; return row
        best_xyz = ranked[0][0]
        row["xtb_optimized"] = True
        xyz_out = xyz_dir / f"{stem}.xyz"
        xyz_out.write_text(best_xyz, encoding="utf-8")
        row["xyz_file"] = str(xyz_out)
        row["__prod_xyz"] = best_xyz          # private: collected by main for the per-chunk merge

        symbols, coords = A.parse_xyz(best_xyz)
        core = find_benzoin_core(symbols, coords)
        if core is None:
            row["error"] = "core_not_found"; return row

        row.update(calc_xtb_product(best_xyz, symbols, coords, core, xtb_bin, mdir / "desc"))
        row.update(calc_morfeus_product(symbols, coords, core))
        if do_multiwfn and mwf_bin:
            row.update(calc_multiwfn_product(best_xyz, symbols, coords, core,
                                             xtb_bin, mwf_bin, mdir / "mwf", stem))

        # ── ΔG = G(product) - G(donor) - G(acceptor) ──────────────────────────
        G_prod, _ = Th.run_ohess(best_xyz, mdir / "prod_ohess", xtb_bin, T, P,
                                 solvent=solvent, cores=cores, timeout=timeout)
        G_prod = Th.parse_xtb_G(G_prod)
        G_d = g_cache.get(Chem.CanonSmiles(donor))
        G_a = g_cache.get(Chem.CanonSmiles(acceptor))
        row["G_donor"], row["G_acceptor"], row["G_product"] = G_d, G_a, G_prod
        if None not in (G_prod, G_d, G_a):
            row["dG_xtb_kcal"] = round((G_prod - G_d - G_a) * HARTREE_TO_KCAL, 4)
        else:
            row["error"] = (row["error"] + ";" if row["error"] else "") + "dG_failed"
        return row
    finally:
        shutil.rmtree(mdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None,
                    help="CSV with donor_smiles,acceptor_smiles[,donor_id,acceptor_id]")
    ap.add_argument("--pair", action="append", metavar="DONOR>>ACCEPTOR",
                    help="explicit pair (repeatable), e.g. 'O=Cc1ccccc1>>O=Cc1ccco1'")
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--xtb-bin", default=shutil.which("xtb") or "/home/schen3/xtb/bin/xtb")
    ap.add_argument("--multiwfn-bin", default=None)
    ap.add_argument("--multiwfn", action="store_true")
    ap.add_argument("--solvent", default="dmso")
    ap.add_argument("--n-confs", type=int, default=10)
    ap.add_argument("--xtb-cores", type=int, default=2)
    ap.add_argument("--parallel-jobs", type=int, default=4)
    ap.add_argument("--ohess-timeout", type=int, default=900)
    ap.add_argument("--T", type=float, default=298.15)
    ap.add_argument("--P", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--conformer", choices=["funnel_v3", "rank"], default="funnel_v3",
                    help="conformer search: funnel_v3 (topology-guarded, default) or plain rank")
    ap.add_argument("--emit-aldehydes", action="store_true",
                    help="also featurize+save each unique aldehyde on its funnel_v3 geometry "
                         "(descriptors + G), so aldehyde and product use the same method")
    ap.add_argument("--ald-output", default=None,
                    help="aldehyde feature CSV (default: <output dir>/aldehydes.csv)")
    ap.add_argument("--xyz-dir", default=None,
                    help="base dir for saved reactant/product xyz (subdirs xyz/ + ald_xyz/). "
                         "Default: <output dir> (SHARED FS). At library scale point this at "
                         "node-local scratch (e.g. $SCRATCH) so geometries are discarded with "
                         "the job and don't bloat shared-FS inodes — funnel_v3 is deterministic "
                         "so any geometry is reproducible from SMILES on demand.")
    ap.add_argument("--xyz-merge-dir", default=None,
                    help="if set, write per-chunk CONSOLIDATED geometries here: one multi-frame "
                         "<output_stem>_reactant.xyz and <output_stem>_product.xyz (each frame "
                         "titled with index/SMILES). 2 files/chunk instead of 1/molecule — keeps "
                         "geometries for later use without per-molecule inode bloat.")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    xtb_bin = shutil.which(args.xtb_bin) or args.xtb_bin
    solvent = "" if args.solvent.lower() == "none" else args.solvent
    do_multiwfn = args.multiwfn and bool(args.multiwfn_bin)

    if args.pair:
        records = []
        for i, s in enumerate(args.pair):
            d, _, a = s.partition(">>")
            records.append({"index": str(i), "donor_smiles": d.strip(),
                            "acceptor_smiles": a.strip()})
    elif args.input:
        with open(args.input, encoding="utf-8") as fh:
            records = list(csv.DictReader(fh))
    else:
        log.error("provide --input or --pair"); return 1
    if args.max:
        records = records[: args.max]
    log.info("Featurizing %d pairs (multiwfn=%s)", len(records), do_multiwfn)

    work_dir = Path(args.work_dir or os.environ.get("TMPDIR", "/tmp")) / "featurize_product"
    # xyz base: node-local scratch when --xyz-dir is set (discarded with the job), else the
    # shared output dir (back-compat). funnel_v3 is deterministic → geometries are reproducible
    # from SMILES, so discarding them costs nothing but avoids per-molecule inode bloat at scale.
    xyz_base = Path(args.xyz_dir) if args.xyz_dir else Path(args.output).parent
    xyz_dir = xyz_base / "xyz"
    for dpath in (work_dir, xyz_dir, Path(args.output).parent):
        dpath.mkdir(parents=True, exist_ok=True)

    # ── 1. per-aldehyde G (computed once per unique aldehyde) ──────────────
    uniq = sorted({Chem.CanonSmiles(r[k]) for r in records
                   for k in ("donor_smiles", "acceptor_smiles")
                   if r.get(k) and Chem.MolFromSmiles(r[k])})
    g_cache: dict[str, float | None] = {}
    ald_xyz_map: dict[str, str | None] = {}   # unique-aldehyde geometry, for per-chunk merge
    if args.emit_aldehydes:
        # Full aldehyde featurization on the funnel_v3 geometry (consistent with products),
        # saving xyz + descriptors + G. One row per unique aldehyde.
        ald_xyz_dir = xyz_base / "ald_xyz"
        ald_xyz_dir.mkdir(parents=True, exist_ok=True)
        ald_out = Path(args.ald_output or Path(args.output).parent / "aldehydes.csv")
        log.info("Featurizing %d unique aldehydes (funnel_v3, workers=%d) -> %s",
                 len(uniq), args.workers, ald_out)
        akw = dict(work_dir=work_dir, ald_xyz_dir=ald_xyz_dir, xtb_bin=xtb_bin,
                   mwf_bin=args.multiwfn_bin, do_multiwfn=do_multiwfn, solvent=solvent,
                   n_confs=args.n_confs, T=args.T, P=args.P, cores=args.xtb_cores,
                   jobs=args.parallel_jobs, timeout=args.ohess_timeout, conformer=args.conformer)
        with open(ald_out, "w", newline="", encoding="utf-8") as afh:
            aw = csv.DictWriter(afh, fieldnames=ALD_FIELDS, extrasaction="ignore")
            aw.writeheader()
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(featurize_aldehyde, smi, i, **akw): smi
                            for i, smi in enumerate(uniq)}
                    for done, fut in enumerate(as_completed(futs), 1):
                        smi = futs[fut]
                        try:
                            row, G = fut.result()
                        except Exception as exc:
                            row, G = {"SMILES": smi, "error": f"exception:{exc}"}, None
                        aw.writerow(row); afh.flush(); g_cache[smi] = G
                        if done % 25 == 0 or done == len(uniq):
                            log.info("  ald %d/%d", done, len(uniq))
            else:
                for i, smi in enumerate(uniq):
                    row, G = featurize_aldehyde(smi, i, **akw)
                    aw.writerow(row); afh.flush(); g_cache[smi] = G
    else:
        log.info("Computing G for %d unique aldehydes (workers=%d) ...", len(uniq), args.workers)
        g_args = (xtb_bin, solvent, args.n_confs, args.T, args.P, args.xtb_cores,
                  args.parallel_jobs, args.ohess_timeout, args.conformer)
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(ald_free_energy, smi, work_dir / f"ald_{i:05d}", *g_args): smi
                        for i, smi in enumerate(uniq)}
                for done, fut in enumerate(as_completed(futs), 1):
                    smi = futs[fut]
                    try:
                        g_cache[smi], ald_xyz_map[smi] = fut.result()
                    except Exception as exc:
                        log.error("ald G failed: %s", exc); g_cache[smi] = None
                    if done % 25 == 0 or done == len(uniq):
                        log.info("  ald G %d/%d", done, len(uniq))
        else:
            for i, smi in enumerate(uniq):
                g_cache[smi], ald_xyz_map[smi] = ald_free_energy(smi, work_dir / f"ald_{i:05d}", *g_args)
                log.info("  ald %d/%d G=%s", i + 1, len(uniq),
                         f"{g_cache[smi]:.6f}" if g_cache[smi] is not None else "FAILED")

    kw = dict(g_cache=g_cache, work_dir=work_dir, xyz_dir=xyz_dir, xtb_bin=xtb_bin,
              mwf_bin=args.multiwfn_bin, do_multiwfn=do_multiwfn, solvent=solvent,
              n_confs=args.n_confs, T=args.T, P=args.P, cores=args.xtb_cores,
              jobs=args.parallel_jobs, timeout=args.ohess_timeout,
              conformer=args.conformer)

    # ── 2. product featurization ──────────────────────────────────────────
    n_ok = n_dg = n_err = 0
    prod_frames: list[tuple] = []          # (index, product_smiles, xyz) for the per-chunk merge
    try:
        fh = open(args.output, "w", newline="", encoding="utf-8")
    except OSError as e:
        # Disk quota exceeded or permission errors: provide guidance for recovery
        if e.errno == 122:  # Disk quota exceeded
            log.error("DISK QUOTA EXCEEDED — cannot write %s. Free up disk space and re-run.",
                      args.output)
        else:
            log.error("Cannot write output %s: %s", args.output, e)
        return 1
    
    with fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()

        def emit(row):
            nonlocal n_ok, n_dg, n_err
            xyz = row.pop("__prod_xyz", None)
            if xyz:
                prod_frames.append((row.get("index"), row.get("product_smiles"), xyz))
            try:
                w.writerow(row); fh.flush()
            except OSError as e:
                if e.errno == 122:  # Disk quota exceeded during writing
                    log.error("DISK QUOTA during writing — temp files left in: %s", args.work_dir)
                raise
            n_ok += 1
            n_dg += row.get("dG_xtb_kcal") is not None
            n_err += bool(row.get("error"))

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(featurize_pair, r, i, **kw): i
                        for i, r in enumerate(records)}
                for fut in as_completed(futs):
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {"index": str(futs[fut]), "error": f"exception:{exc}"}
                    emit(row)
                    log.info("idx=%s type=%s dG=%s err=%s", row.get("index"),
                             row.get("reaction_type"), row.get("dG_xtb_kcal"),
                             row.get("error") or "")
        else:
            for i, r in enumerate(records):
                try:
                    row = featurize_pair(r, i, **kw)
                except Exception as exc:
                    row = {"index": str(i), "error": f"exception:{exc}"}
                emit(row)
                log.info("idx=%s type=%s dG=%s err=%s", row.get("index"),
                         row.get("reaction_type"), row.get("dG_xtb_kcal"),
                         row.get("error") or "")

    log.info("Wrote %s — %d rows, %d with dG, %d errors", args.output, n_ok, n_dg, n_err)

    # ── per-chunk consolidated geometries (one multi-frame xyz each) ──────────
    # Reactant + product geometries merged into 2 files per chunk (not 1 file/molecule):
    # bounded inode count, each frame titled for later lookup, written at job END so an
    # interrupted chunk loses only its (re-runnable) geometries, not prior chunks.
    if args.xyz_merge_dir:
        md = Path(args.xyz_merge_dir); md.mkdir(parents=True, exist_ok=True)
        stem = Path(args.output).stem

        def _frame(xyz_block, title):
            lines = (xyz_block or "").splitlines()
            if len(lines) >= 2:
                lines[1] = title
            return "\n".join(lines) + "\n"

        pf = md / f"{stem}_product.xyz"
        with open(pf, "w", encoding="utf-8") as f:
            for index, smi, xyz in prod_frames:
                f.write(_frame(xyz, f"index={index} product={smi}"))
        rf = md / f"{stem}_reactant.xyz"
        with open(rf, "w", encoding="utf-8") as f:
            for smi, xyz in ald_xyz_map.items():
                if xyz:
                    f.write(_frame(xyz, f"reactant={smi}"))
        log.info("Merged geometries: %d products -> %s, %d reactants -> %s",
                 len(prod_frames), pf.name, sum(1 for v in ald_xyz_map.values() if v), rf.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
