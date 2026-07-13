#!/usr/bin/env python3
"""
Benzoin Thermochemistry — ΔG Calculator (ORCA variant)
=======================================================
For each (aldehyde, benzoin) pair (from CSV or --test / --smiles):

  1. Conformer search  ETKDGv3 → MMFF → GFN2-xTB --opt --alpb <solvent>  (--n-confs N)
  2. GFN2-xTB --ohess tight --alpb <solvent>  on the lowest-energy conformer
     → G = E_el(ALPB) + G(RRHO)
  3. [Optional] Shermo quasi-RRHO correction on g98.out
     → G = E_el(ALPB) + G(qRRHO)
  4. [Optional] ORCA DFT single-point + CPCM (--sp-method PBE0-D4)
     → G = E_el(ORCA+CPCM) + G(RRHO)_xtb     [or + G(qRRHO)_shermo]
  5. ΔG = G_bz − 2·G_ald  (all variants)

Reaction: 2 R-CHO → R-CH(OH)-C(=O)-R

Usage
-----
  python benzoin_orca.py --test                           # xTB + Shermo only
  python benzoin_orca.py --test --sp-method PBE0-D4       # + ORCA SP
  python benzoin_orca.py --input chunk_00000.csv --workers 32 --sp-method PBE0-D4

Dependencies
------------
  pip install rdkit tqdm matplotlib
  xtb >= 6.4  (in PATH or --xtb-bin)
  Shermo      (--shermo-bin, optional)
  ORCA 6.x    (--orca-bin, optional; serial, no MPI required)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs

import conf_funnel_v3   # funnel_v3 ranker (same conformer method as the xTB screen)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Default tool paths ─────────────────────────────────────────────────────────
_DEFAULT_ORCA   = "/home/schen3/orca/orca"
_DEFAULT_SHERMO = "/home/schen3/.local/bin/Shermo"

# ── Physical constants ─────────────────────────────────────────────────────────
HARTREE_TO_KCAL = 627.509474
KCAL_TO_KJ      = 4.184

# ── Benzoin condensation SMARTS ────────────────────────────────────────────────
# Couple ONLY carbon-bound aldehydes ([CX3H1]=O with a carbon neighbour). The
# unconstrained "[CX3H1:1](=[O:2])" also matches N-/O-formyl (formamide / formate),
# so on substrates bearing both a real R-CHO and an N/O-formyl the reaction could
# couple at the wrong site and leave the true CHO free in the product
# (see memory benzoin-generator-formyl-bug; check_smiles.py flags the survivors).
_BENZOIN_RXN = AllChem.ReactionFromSmarts(
    "[CX3H1:1](=[O:2])[#6:5].[CX3H1:3](=[O:4])[#6:6]"
    ">>[C:5][C:1]([OH1:2])[C:3](=[O:4])[C:6]"
)

# ── Test aldehydes ─────────────────────────────────────────────────────────────
_TEST_ALDEHYDES: list[tuple[str, str]] = [
    ("benzaldehyde",              "O=Cc1ccccc1"),
    ("4-methylbenzaldehyde",      "O=Cc1ccc(C)cc1"),
    ("4-methoxybenzaldehyde",     "O=Cc1ccc(OC)cc1"),
    ("4-chlorobenzaldehyde",      "O=Cc1ccc(Cl)cc1"),
    ("4-nitrobenzaldehyde",       "O=Cc1ccc([N+](=O)[O-])cc1"),
    ("furfural",                  "O=Cc1ccco1"),
    ("thiophene-2-carbaldehyde",  "O=Cc1cccs1"),
    ("cinnamaldehyde",            "O=C/C=C/c1ccccc1"),
    ("hexanal",                   "O=CCCCCC"),
    ("pivalaldehyde",             "O=CC(C)(C)C"),
]

# ── Solvent tables ─────────────────────────────────────────────────────────────
_SOLVENT_EPS: dict[str, float] = {
    "dmso": 46.7,
}
# ORCA-recognised solvent names (for CPCM keyword)
_ORCA_SOLVENT: dict[str, str] = {
    "dmso": "DMSO",
}

# ── Output CSV columns ────────────────────────────────────────────────────────
_CSV_FIELDS = [
    "idx", "index", "PubChem_CID", "aldehyde_name", "aldehyde_smiles", "benzoin_smiles",
    "dG_xtb_kcal",
    "dG_shermo_kcal",
    "dG_orca_kcal",
    "dG_orca_shermo_kcal",
    # component free energies (Eh): reactant aldehyde and product benzoin
    "G_ald_xtb_Eh", "G_bz_xtb_Eh",        # xTB-ohess Gibbs free energy
    "G_ald_orca_Eh", "G_bz_orca_Eh",      # DFT single-point + xTB-RRHO composite G
    "E_ald_orca_Eh", "E_bz_orca_Eh",      # bare DFT single-point electronic energy
    "benzoin_xyz_file",                    # saved product geometry
    "error",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Conformer generation
# ══════════════════════════════════════════════════════════════════════════════

def _auto_nconfs(rotbonds: int, n_confs_max: int) -> int:
    if rotbonds <= 3:
        return 1
    frac = min(1.0, (rotbonds - 3) / 6.0)
    return min(n_confs_max, max(3, round(n_confs_max * frac)))


def _mol_rotbonds(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcNumRotatableBonds(mol) if mol else 0


def _smiles_to_xyz_conformers(smiles: str, n_confs: int, title: str = "") -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    params.maxIterations = 1000
    n_emb = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if n_emb == 0:
        rc = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if rc != 0:
            return []
    try:
        MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
    except Exception:
        pass
    xyz_list: list[str] = []
    for conf in mol.GetConformers():
        n = mol.GetNumAtoms()
        lines = [str(n), title or smiles[:60]]
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol()}  {p.x:.8f}  {p.y:.8f}  {p.z:.8f}")
        xyz_list.append("\n".join(lines))
    return xyz_list


# ══════════════════════════════════════════════════════════════════════════════
#  xTB helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run_xtb(cmd: list[str], work_dir: Path, timeout: int = 600) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            cmd, cwd=str(work_dir),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.debug("xtb timed out: %s", " ".join(cmd[:4]))
    except Exception as exc:
        log.debug("xtb error: %s", exc)
    return None


_ENERGY_PATS = [
    r"::\s*total energy\s+([-\d.]+)\s*Eh\s*::",
    r"total energy\s+([-\d.]+)\s*Eh",
    r"TOTAL ENERGY\s+([-\d.]+)",
]


def _parse_xtb_energy(stdout: str) -> float | None:
    """Return the LAST total energy (Eh) — final optimised value."""
    last: float | None = None
    for line in stdout.splitlines():
        for pat in _ENERGY_PATS:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                try:
                    last = float(m.group(1))
                except ValueError:
                    pass
                break
    return last


def _xtb_opt_energy(xyz_str: str, work_dir: Path, xtb_bin: str,
                    charge: int = 0, solvent: str = "",
                    cores: int = 1) -> tuple[str | None, float | None]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "input.xyz").write_text(xyz_str, encoding="utf-8")
    cmd = [xtb_bin, "input.xyz", "--gfn", "2", "--opt", "tight",
           "--charge", str(charge), "--norestart", "--parallel", str(cores)]
    if solvent:
        cmd += ["--alpb", solvent]
    r = _run_xtb(cmd, work_dir, timeout=600)
    opt = work_dir / "xtbopt.xyz"
    if not opt.exists():
        return None, None
    energy = _parse_xtb_energy(r.stdout) if r else None
    return opt.read_text(encoding="utf-8"), energy


def _pick_best_conformer(
    xyz_strings: list[str], work_dir: Path, xtb_bin: str, solvent: str = ""
) -> tuple[str | None, int]:
    if not xyz_strings:
        return None, 0
    if len(xyz_strings) == 1:
        opt_xyz, _ = _xtb_opt_energy(xyz_strings[0], work_dir / "c000", xtb_bin, solvent=solvent)
        return opt_xyz or xyz_strings[0], (1 if opt_xyz else 0)
    best_xyz = xyz_strings[0]
    best_E   = float("inf")
    n_ok     = 0
    for i, xyz in enumerate(xyz_strings):
        opt_xyz, E = _xtb_opt_energy(xyz, work_dir / f"c{i:03d}", xtb_bin, solvent=solvent)
        if opt_xyz is None:
            continue
        n_ok += 1
        if E is not None and E < best_E:
            best_E, best_xyz = E, opt_xyz
        elif E is None and best_E == float("inf"):
            best_xyz = opt_xyz
    return best_xyz, n_ok


def _build_best_xyz(
    smiles: str, work_dir: Path, xtb_bin: str,
    n_confs_max: int, title: str = "", solvent: str = "",
) -> tuple[str | None, int]:
    rb = _mol_rotbonds(smiles)
    n  = _auto_nconfs(rb, n_confs_max)
    log.debug("  rotbonds=%d → %d conformer(s) for '%s'", rb, n, title)
    xyz_list = _smiles_to_xyz_conformers(smiles, n, title)
    if not xyz_list:
        return None, 0
    best, _ = _pick_best_conformer(xyz_list, work_dir / "conf", xtb_bin, solvent=solvent)
    return best, len(xyz_list)


# Boltzmann constant in Hartree/K → RT at the run temperature.
_KB_HARTREE = 3.166811563e-6


def _ensemble_free_energy(Gs: list[float], T: float) -> float | None:
    """Boltzmann ensemble free energy (Hartree):  G = −RT ln Σ exp(−(G_i−G0)/RT) + G0.
    Down-weights high-energy conformers; equals min(G) when one conformer dominates."""
    if not Gs:
        return None
    RT = _KB_HARTREE * T
    G0 = min(Gs)
    Z = sum(math.exp(-(g - G0) / RT) for g in Gs)
    return G0 - RT * math.log(Z)


def _rank_conformers(
    smiles: str, work_dir: Path, xtb_bin: str,
    n_confs_max: int, title: str = "", solvent: str = "",
    cores: int = 1, workers: int = 1,
) -> list[tuple[str, float]]:
    """xTB-optimise all conformers and return [(opt_xyz, E_xtb)] sorted by energy.
    `cores` = threads per xTB; `workers` = conformers optimised concurrently
    (workers x cores should fit the task allocation)."""
    rb = _mol_rotbonds(smiles)
    n = _auto_nconfs(rb, n_confs_max)
    xyz_list = _smiles_to_xyz_conformers(smiles, n, title)
    out: list[tuple[str, float]] = []

    def _opt(i_xyz):
        i, xyz = i_xyz
        return _xtb_opt_energy(xyz, work_dir / f"c{i:03d}", xtb_bin,
                               solvent=solvent, cores=cores)

    if workers > 1 and len(xyz_list) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_opt, enumerate(xyz_list)))
    else:
        res = [_opt(t) for t in enumerate(xyz_list)]
    for opt_xyz, E in res:
        if opt_xyz is not None and E is not None:
            out.append((opt_xyz, E))
    out.sort(key=lambda t: t[1])
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  xTB --ohess runner and parser
# ══════════════════════════════════════════════════════════════════════════════

def _write_xtb_inp(path: Path, T: float, P_bar: float) -> None:
    path.write_text(
        f"$thermo\n   temp={T:.4f}\n   press={P_bar * 1e5:.2f}\n$end\n",
        encoding="utf-8",
    )


def run_ohess(
    xyz_str: str,
    work_dir: Path,
    xtb_bin: str,
    T: float = 298.15,
    P_atm: float = 1.0,
    solvent: str = "",
    timeout: int = 3600,
    cores: int = 1,
) -> tuple[str, Path | None]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "mol.xyz").write_text(xyz_str, encoding="utf-8")
    _write_xtb_inp(work_dir / "xtb.inp", T, P_atm * 1.01325)
    cmd = [xtb_bin, "mol.xyz", "--gfn", "2", "--ohess", "tight",
           "--input", "xtb.inp", "--norestart", "--parallel", str(cores)]
    if solvent:
        cmd += ["--alpb", solvent]
    stdout = ""
    try:
        r = subprocess.run(cmd, cwd=str(work_dir),
                           capture_output=True, text=True, timeout=timeout)
        stdout = r.stdout + r.stderr
        if r.returncode != 0:
            log.debug("xtb ohess exit %d  in %s", r.returncode, work_dir)
    except subprocess.TimeoutExpired:
        log.warning("xtb ohess timed out (%.0f s)  in %s", timeout, work_dir)
    except Exception as exc:
        log.warning("xtb ohess error: %s", exc)
    g98 = work_dir / "g98.out"
    return stdout, (g98 if g98.exists() else None)


def parse_xtb_G(stdout: str) -> float | None:
    for pat in [
        r"::\s*total free energy\s+([-\d.]+)\s*Eh\s*::",
        r"::\s*G\s*\(total\)\s+([-\d.]+)\s*::",
        r"::\s*G\s*\(T\)\s+([-\d.]+)\s*::",
        r"total free energy\s+([-\d.]+)\s*Eh",
        r"G\(total\)\s+([-\d.]+)\s*(?:Eh|au)",
        r"Gibbs free energy.*?=\s*([-\d.]+)\s*(?:Eh|au)",
        r"G\(T\)\s*=\s*([-\d.]+)\s*(?:Eh|au|Hartree)",
    ]:
        m = re.search(pat, stdout, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    # Fallback: E_el (last occurrence) + G(RRHO) contribution
    E_el = G_rrho = None
    for line in stdout.splitlines():
        for p in _ENERGY_PATS:
            m = re.search(p, line, re.IGNORECASE)
            if m:
                try: E_el = float(m.group(1))
                except ValueError: pass
                break
        if G_rrho is None:
            for p in [r"::\s*G\(RRHO\)\s+contrib\.\s+([-\d.]+)\s*Eh\s*::",
                      r"G\(RRHO\)\s+contrib\.\s+([-\d.]+)\s*(?:Eh|au)"]:
                m = re.search(p, line, re.IGNORECASE)
                if m:
                    try: G_rrho = float(m.group(1)); break
                    except ValueError: pass
    if E_el is not None and G_rrho is not None:
        return E_el + G_rrho
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Shermo runner and parser
# ══════════════════════════════════════════════════════════════════════════════

def run_shermo(
    g98_path: Path,
    shermo_bin: str,
    E_el: float | None = None,
    T: float = 298.15,
    P_atm: float = 1.0,
    freq_cutoff: float = 100.0,
    timeout: int = 120,
) -> str:
    work_dir = g98_path.parent
    e_str = f"{E_el:.10f}" if E_el is not None else "0"
    ini = (
        f"E= {e_str}\n"
        f"conc= 0\n"
        f"ilowfreq= 2\n"
        f"intpvib= {freq_cutoff:.1f}\n"
        f"T= {T:.4f}\n"
        f"P= {P_atm:.4f}\n"
    )
    (work_dir / "settings.ini").write_text(ini, encoding="utf-8")
    try:
        r = subprocess.run(
            [shermo_bin, g98_path.name],
            cwd=str(work_dir),
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout + r.stderr
    except Exception as exc:
        log.debug("Shermo failed: %s", exc)
    return ""


def parse_shermo_G(stdout: str) -> float | None:
    m = re.search(
        r"Sum of electronic energy and thermal correction to G:\s+([-\d.]+)\s*a\.u\.",
        stdout,
    )
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  ORCA single-point engine
# ══════════════════════════════════════════════════════════════════════════════

def _build_orca_keywords(method: str) -> str:
    """Convert 'PBE0-D4' → 'PBE0 D4', 'B3LYP-D3BJ' → 'B3LYP D3BJ', etc."""
    u = method.upper()
    disp = ""
    for suffix, kw in [("-D4", "D4"), ("-D3BJ", "D3BJ"), ("-D3", "D3")]:
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            disp = kw
            break
    return f"{u} {disp}".strip() if disp else u


def _build_orca_input(
    method: str,
    basis: str,
    orca_solvent: str,
    nprocs: int,
    maxcore_mb: int,
) -> str:
    solv_kw = f"CPCM({orca_solvent})" if orca_solvent else ""

    # Composite "-3c" methods (r2SCAN-3c, wB97X-3c, B97-3c, ...) carry their own
    # basis + dispersion + geometric counterpoise, so no separate basis/RI keywords.
    # SCF convergence: default TightSCF (overkill); set ORCA_SCF=default/omit to use ORCA's
    # own default (normal-level for r2SCAN-3c SP; ~20-40% faster, <0.1 kcal on relative ΔG).
    scf = os.environ.get("ORCA_SCF", "TightSCF")
    scf_kw = [] if scf.lower() in ("", "default", "omit", "none") else [scf]
    if method.lower().endswith("-3c"):
        kw_parts = [method, *scf_kw, "NoMOPrint"]
    else:
        func_kw = _build_orca_keywords(method)
        kw_parts = [func_kw, basis, "RIJCOSX", "def2/J", *scf_kw, "NoMOPrint"]
    if solv_kw:
        kw_parts.append(solv_kw)

    lines = [f"! {' '.join(kw_parts)}", ""]
    if nprocs > 1:
        lines += [f"%pal nprocs {nprocs} end", ""]
    lines += [f"%maxcore {maxcore_mb}", "", "* xyzfile 0 1 mol.xyz", ""]
    return "\n".join(lines)


def _parse_orca_energy(output: str) -> float | None:
    """Extract the last 'FINAL SINGLE POINT ENERGY' from ORCA output (Eh)."""
    last: float | None = None
    for line in output.splitlines():
        m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", line)
        if m:
            try:
                last = float(m.group(1))
            except ValueError:
                pass
    return last


def calc_orca_sp(
    xyz_path: Path,
    method: str,
    basis: str,
    orca_solvent: str,
    charge: int = 0,
    nprocs: int = 1,
    maxcore_mb: int = 1792,
    orca_bin: str = _DEFAULT_ORCA,
    timeout: int = 7200,
) -> float | None:
    """ORCA DFT + CPCM single-point; returns E_total (Eh) or None."""
    if not xyz_path.exists():
        return None

    work_dir = xyz_path.parent / "orca_sp"
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(xyz_path, work_dir / "mol.xyz")

    inp = _build_orca_input(method, basis, orca_solvent, nprocs, maxcore_mb)
    (work_dir / "input.inp").write_text(inp, encoding="utf-8")

    try:
        env = {**os.environ,
               "PATH": f"{Path(orca_bin).parent}:{os.environ.get('PATH', '')}",
               "LD_LIBRARY_PATH": (
                   f"{Path(orca_bin).parent / 'lib'}:"
                   f"{os.environ.get('LD_LIBRARY_PATH', '')}"
               )}
        r = subprocess.run(
            [orca_bin, "input.inp"],
            cwd=str(work_dir),
            capture_output=True, text=True, timeout=timeout,
            env=env,
        )
        output = r.stdout + r.stderr
        # Retain the human-readable ORCA log alongside input.inp / mol.xyz.
        try:
            (work_dir / "input.out").write_text(output, encoding="utf-8")
        except Exception:
            pass
        E = _parse_orca_energy(output)
        if E is None:
            log.debug("ORCA parse failed in %s (exit=%d)\n%.300s",
                      work_dir, r.returncode, output)
        return E
    except subprocess.TimeoutExpired:
        log.warning("ORCA timed out (%s s) in %s", timeout, work_dir)
    except Exception as exc:
        log.debug("ORCA error: %s", exc)
    finally:
        # Remove large binary scratch files
        for pattern in ["*.gbw", "*.tmp", "*.densities", "*.ges", "*_ges"]:
            for f in work_dir.glob(pattern):
                try:
                    f.unlink()
                except Exception:
                    pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Per-molecule workflow
# ══════════════════════════════════════════════════════════════════════════════

def _make_benzoin_smiles(ald_smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(ald_smiles)
    if mol is None:
        return None
    try:
        products = _BENZOIN_RXN.RunReactants((mol, mol))
    except Exception:
        return None
    for prod_tuple in products:
        try:
            p = prod_tuple[0]
            Chem.SanitizeMol(p)
            return Chem.MolToSmiles(p)
        except Exception:
            continue
    return None


def _dG_kcal(G_prod: float | None, G_react: float | None) -> float | None:
    if G_prod is None or G_react is None:
        return None
    return round((G_prod - 2.0 * G_react) * HARTREE_TO_KCAL, 4)


_MolResult = dict[str, Any]


def _too_big(geom: Path, max_atoms: int) -> bool:
    if max_atoms <= 0 or not geom.exists():
        return False
    lines = geom.read_text().splitlines()
    n_heavy = sum(1 for l in lines[2:] if l.split() and l.split()[0] != "H")
    return n_heavy > max_atoms


def _ohess_orca(
    xyz: str, work_dir: Path, *, xtb_bin, T, P_atm, solvent, sp_method, sp_basis,
    orca_bin, orca_nprocs, orca_maxcore, orca_solvent, max_atoms, xtb_cores=1,
) -> tuple[float | None, float | None, float | None, Path | None]:
    """One conformer: ohess → (G_xtb, E_el_xtb), then ORCA SP → E_orca."""
    stdout, g98 = run_ohess(xyz, work_dir, xtb_bin, T, P_atm, solvent=solvent,
                            cores=xtb_cores)
    G = parse_xtb_G(stdout)
    E_el = _parse_xtb_energy(stdout)
    E_orca = None
    if G is not None and sp_method:
        geom = work_dir / "xtbopt.xyz"
        if not geom.exists():
            geom = work_dir / "mol.xyz"
        if not _too_big(geom, max_atoms):
            E_orca = calc_orca_sp(geom, sp_method, sp_basis, orca_solvent,
                                  nprocs=orca_nprocs, maxcore_mb=orca_maxcore,
                                  orca_bin=orca_bin)
    return G, E_el, E_orca, g98


def _mol_pipeline(
    smiles: str,
    work_dir: Path,
    *,
    label: str,
    xtb_bin: str,
    shermo_bin: str | None,
    T: float,
    P_atm: float,
    n_confs_max: int,
    solvent: str,
    freq_cutoff: float,
    sp_method: str | None,
    sp_basis: str,
    orca_bin: str,
    orca_nprocs: int,
    orca_maxcore: int,
    orca_solvent: str,
    max_atoms: int,
    ensemble_k: int = 1,
    ensemble_window: float = 3.0,
    conformer: str = "default",
) -> _MolResult:
    # "G_orca" = composite free energy E_orca + xTB-RRHO thermal (single conformer
    # OR Boltzmann ensemble); it is the DFT-level ΔG term calc_thermo_one consumes.
    res: _MolResult = {"G": None, "E_el": None, "G_sh": None,
                       "E_orca": None, "G_orca": None, "xyz": None, "errors": []}

    kw = dict(xtb_bin=xtb_bin, T=T, P_atm=P_atm, solvent=solvent,
              sp_method=sp_method, sp_basis=sp_basis, orca_bin=orca_bin,
              orca_nprocs=orca_nprocs, orca_maxcore=orca_maxcore,
              orca_solvent=orca_solvent, max_atoms=max_atoms)

    # Conformer ranking: use the SAME method as the xTB screen (funnel_v3) so the
    # reactant and product geometries are consistent with the completed simulation.
    def _get_ranked(wd):
        if conformer == "funnel_v3":
            return conf_funnel_v3.rank_conformers_funnel_v3(
                smiles, wd, xtb_bin, n_confs_max=n_confs_max, title=label,
                solvent=solvent, cores=1, workers=1)
        return _rank_conformers(smiles, wd, xtb_bin, n_confs_max, label, solvent=solvent)

    if ensemble_k > 1:
        # ── Conformer-ensemble path (better DFT label for flexible molecules) ──
        ranked = _get_ranked(work_dir / "rank")
        if not ranked:
            res["errors"].append(f"{label}:embed_failed")
            return res
        E0 = ranked[0][1]
        win_Eh = ensemble_window / HARTREE_TO_KCAL
        sel = [(x, e) for x, e in ranked[:ensemble_k] if e - E0 <= win_Eh]
        comps: list[float] = []          # composite G_orca per conformer
        for i, (xyz, _) in enumerate(sel):
            G, E_el, E_orca, g98 = _ohess_orca(xyz, work_dir / f"ens{i:02d}", **kw)
            if G is None:
                continue
            if res["G"] is None:         # rank-0 conformer → single-conformer dG_xtb
                res["G"], res["E_el"] = G, E_el
                if shermo_bin and g98:
                    res["G_sh"] = parse_shermo_G(
                        run_shermo(g98, shermo_bin, E_el=E_el, T=T, P_atm=P_atm,
                                   freq_cutoff=freq_cutoff))
            if E_orca is not None and E_el is not None:
                comps.append(E_orca + (G - E_el))
        if res["G"] is None:
            res["errors"].append(f"{label}:G_parse_failed")
            return res
        res["G_orca"] = _ensemble_free_energy(comps, T)
        res["xyz"] = sel[0][0] if sel else None
        log.debug("[%s] ensemble: %d/%d conformers, G_orca=%s",
                  label, len(comps), len(sel), res["G_orca"])
        return res

    # ── Single-conformer path (default) ───────────────────────────────────
    ranked = _get_ranked(work_dir / "rank")
    best_xyz = ranked[0][0] if ranked else None
    if best_xyz is None:
        res["errors"].append(f"{label}:embed_failed")
        return res
    G, E_el, E_orca, g98 = _ohess_orca(best_xyz, work_dir / "ohess", **kw)
    res["G"], res["E_el"] = G, E_el
    if G is None:
        res["errors"].append(f"{label}:G_parse_failed")
        return res
    if shermo_bin and g98:
        res["G_sh"] = parse_shermo_G(
            run_shermo(g98, shermo_bin, E_el=E_el, T=T, P_atm=P_atm,
                       freq_cutoff=freq_cutoff))
        if res["G_sh"] is None:
            res["errors"].append(f"{label}:shermo_parse_failed")
    res["E_orca"] = E_orca
    if E_orca is not None and E_el is not None:
        res["G_orca"] = E_orca + (G - E_el)
    res["xyz"] = best_xyz
    return res


def calc_thermo_one(
    rec: dict,
    idx: int,
    work_root: Path,
    xtb_bin: str,
    shermo_bin: str | None,
    T: float,
    P_atm: float,
    n_confs_max: int,
    solvent: str = "dmso",
    freq_cutoff: float = 100.0,
    sp_method: str | None = None,
    sp_basis: str = "def2-TZVP",
    orca_bin: str = _DEFAULT_ORCA,
    orca_nprocs: int = 1,
    orca_maxcore: int = 3000,
    max_atoms: int = 0,
    ensemble_k: int = 1,
    ensemble_window: float = 3.0,
    conformer: str = "default",
) -> dict[str, Any]:
    csv_index   = str(rec.get("index") or "").strip()
    pubchem_cid = str(rec.get("PubChem_CID") or "").strip()
    ald_smi     = (rec.get("aldehyde_smiles") or rec.get("SMILES") or "").strip()
    bz_smi      = (rec.get("benzoin_smiles") or "").strip()
    name        = (rec.get("aldehyde_name") or rec.get("name") or
                   (f"cid{pubchem_cid}" if pubchem_cid else None) or
                   f"idx{csv_index or idx}").strip()

    stem    = re.sub(r"[^\w\-]", "_", name)[:60] or f"mol_{idx:06d}"
    ald_dir = work_root / stem / "ald"
    bz_dir  = work_root / stem / "bz"

    row: dict[str, Any] = {f: None for f in _CSV_FIELDS}
    row.update({"idx": idx, "index": csv_index, "PubChem_CID": pubchem_cid,
                "aldehyde_name": name, "aldehyde_smiles": ald_smi,
                "benzoin_smiles": bz_smi, "error": ""})

    orca_solvent = _ORCA_SOLVENT.get(solvent.lower(), "") if solvent else ""
    pipe_kw = dict(
        xtb_bin=xtb_bin, shermo_bin=shermo_bin,
        T=T, P_atm=P_atm, n_confs_max=n_confs_max,
        solvent=solvent, freq_cutoff=freq_cutoff,
        sp_method=sp_method, sp_basis=sp_basis,
        orca_bin=orca_bin, orca_nprocs=orca_nprocs,
        orca_maxcore=orca_maxcore, orca_solvent=orca_solvent,
        max_atoms=max_atoms, ensemble_k=ensemble_k,
        ensemble_window=ensemble_window, conformer=conformer,
    )

    errors: list[str] = []

    if not bz_smi and ald_smi:
        bz_smi = _make_benzoin_smiles(ald_smi)
        row["benzoin_smiles"] = bz_smi or ""
        if not bz_smi:
            errors.append("benzoin_smarts_failed")

    _empty = {"G": None, "E_el": None, "G_sh": None, "E_orca": None, "G_orca": None}
    if ald_smi:
        ald = _mol_pipeline(ald_smi, ald_dir, label="ald", **pipe_kw)
    else:
        ald = {**_empty, "errors": ["ald:no_smiles"]}

    if bz_smi:
        bz = _mol_pipeline(bz_smi, bz_dir, label="bz", **pipe_kw)
    else:
        bz = {**_empty, "errors": ["bz:no_smiles"]}

    errors += ald["errors"] + bz["errors"]

    row["dG_xtb_kcal"]    = _dG_kcal(bz["G"],    ald["G"])
    row["dG_shermo_kcal"] = _dG_kcal(bz["G_sh"], ald["G_sh"])

    # DFT-level ΔG from the composite free energy (single conformer or ensemble).
    row["dG_orca_kcal"] = _dG_kcal(bz.get("G_orca"), ald.get("G_orca"))
    el_a, el_b = ald["E_el"], bz["E_el"]
    if (ald["G_sh"] is not None and bz["G_sh"] is not None and el_a and el_b
            and ald["E_orca"] is not None and bz["E_orca"] is not None):
        row["dG_orca_shermo_kcal"] = _dG_kcal(
            bz["E_orca"] + (bz["G_sh"]  - el_b),
            ald["E_orca"] + (ald["G_sh"] - el_a),
        )

    # Record the component free energies (reactant aldehyde + product benzoin):
    # xTB-ohess G, DFT-single-point composite G, and bare DFT electronic energy.
    row["G_ald_xtb_Eh"],  row["G_bz_xtb_Eh"]  = ald.get("G"),      bz.get("G")
    row["G_ald_orca_Eh"], row["G_bz_orca_Eh"] = ald.get("G_orca"), bz.get("G_orca")
    row["E_ald_orca_Eh"], row["E_bz_orca_Eh"] = ald.get("E_orca"), bz.get("E_orca")
    # Save the product (benzoin) geometry to a persistent xyz file.
    if bz.get("xyz"):
        try:
            xyz_dir = work_root.parent / "benzoin_xyz"
            xyz_dir.mkdir(parents=True, exist_ok=True)
            bz_path = xyz_dir / f"{stem}.xyz"
            bz_path.write_text(bz["xyz"], encoding="utf-8")
            row["benzoin_xyz_file"] = str(bz_path)
        except Exception as e:
            errors.append(f"bz_xyz_save_failed:{e}")

    row["error"] = "; ".join(errors)

    if row["dG_xtb_kcal"] is not None:
        sp_str = (f"  ΔG(ORCA+RRHO)=%+.2f" % row["dG_orca_kcal"]
                  if row.get("dG_orca_kcal") is not None else "")
        log.info("[%s]  ΔG(xTB)=%+.2f  ΔG(Shermo)=%s%s  kcal/mol",
                 name, row["dG_xtb_kcal"],
                 f"{row['dG_shermo_kcal']:+.2f}"
                 if row.get("dG_shermo_kcal") is not None else "—",
                 sp_str)

    # Free per-molecule scratch immediately (the product xyz already lives in
    # work_root.parent/benzoin_xyz). Bounds peak inode use at full-library scale
    # to ~workers concurrent mol-dirs instead of accumulating all CHUNK of them.
    try:
        shutil.rmtree(work_root / stem, ignore_errors=True)
    except Exception:
        pass

    return row


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _plot_tag(sp_method: str | None, sp_basis: str, solvent: str) -> str:
    method = f"_{sp_method}_{sp_basis}".replace("/", "-") if sp_method else "_xTB"
    solv   = f"_{solvent}" if solvent else "_gas"
    return f"{method}{solv}"


def plot_results(
    results: list[dict],
    out_dir: Path,
    shermo_bin: str | None,
    sp_method: str | None,
    sp_basis: str = "",
    solvent: str = "",
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        log.warning("matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    rows = [r for r in results if r.get("dG_xtb_kcal") is not None]
    if not rows:
        return

    tag   = _plot_tag(sp_method, sp_basis, solvent)
    names = [r["aldehyde_name"] for r in rows]
    n     = len(names)
    y     = list(range(n))

    COLORS = {
        "xtb":    "#4C72B0",
        "shermo": "#DD8452",
        "orca":   "#55A868",
        "orcash": "#C44E52",
    }

    _candidate_levels = [
        ("dG_xtb_kcal",          "xTB ohess+ALPB",                           COLORS["xtb"]),
        ("dG_shermo_kcal",       "xTB + Shermo qRRHO",                       COLORS["shermo"]),
        ("dG_orca_kcal",         f"ORCA {sp_method}/CPCM + xTB RRHO",        COLORS["orca"]),
        ("dG_orca_shermo_kcal",  f"ORCA {sp_method}/CPCM + Shermo qRRHO",    COLORS["orcash"]),
    ]

    def _level_active(field: str) -> bool:
        if "shermo" in field and not shermo_bin:
            return False
        if "orca" in field and not sp_method:
            return False
        return any(r.get(field) is not None for r in rows)

    levels  = [(f, lbl, col) for f, lbl, col in _candidate_levels if _level_active(f)]
    nl      = len(levels)
    bh      = min(0.18, 0.7 / max(nl, 1))
    offsets = [i - (nl - 1) / 2 for i in range(nl)]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * n)))
    for (field, label, color), off in zip(levels, offsets):
        pts = [(yi + off * bh, r[field]) for yi, r in zip(y, rows)
               if r.get(field) is not None]
        if not pts:
            continue
        ys, vals = zip(*pts)
        bars = ax.barh(list(ys), list(vals), height=bh,
                       color=color, label=label, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(v + (0.15 if v >= 0 else -0.15), bar.get_y() + bar.get_height() / 2,
                    f"{v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7, color="dimgray")

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ΔG  (kcal mol⁻¹)", fontsize=10)
    method_str  = f" — {sp_method}/{sp_basis}" if sp_method else ""
    solvent_str = f" in {solvent}" if solvent else ""
    ax.set_title(f"Benzoin condensation  2 R-CHO → R-CH(OH)-C(=O)-R"
                 f"{method_str}{solvent_str}\n"
                 f"negative ΔG = thermodynamically favourable", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", which="major", linestyle=":", alpha=0.4)
    plt.tight_layout()
    bar_png = out_dir / f"delta_G{tag}.png"
    plt.savefig(bar_png, dpi=150)
    plt.close()
    log.info("Bar chart saved : %s", bar_png)

    orca_rows = [r for r in rows if r.get("dG_orca_kcal") is not None]
    if len(orca_rows) < 2:
        return

    xtb_vals  = [r["dG_xtb_kcal"]  for r in orca_rows]
    orca_vals = [r["dG_orca_kcal"] for r in orca_rows]
    pnames    = [r["aldehyde_name"] for r in orca_rows]

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(xtb_vals, orca_vals, color=COLORS["orca"], s=60, zorder=3)
    for x, y_, name in zip(xtb_vals, orca_vals, pnames):
        ax2.annotate(name, (x, y_), textcoords="offset points", xytext=(5, 3),
                     fontsize=7, color="dimgray")
    all_vals = xtb_vals + orca_vals
    lo, hi = min(all_vals) - 1, max(all_vals) + 1
    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y = x")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax2.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_xlabel("ΔG(xTB ohess+ALPB)  (kcal mol⁻¹)", fontsize=10)
    ax2.set_ylabel(f"ΔG(ORCA {sp_method}+xTB RRHO)  (kcal mol⁻¹)", fontsize=10)
    ax2.set_title("Theory-level comparison", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    cmp_png = out_dir / f"delta_G{tag}_compare.png"
    plt.savefig(cmp_png, dpi=150)
    plt.close()
    log.info("Comparison plot saved: %s", cmp_png)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benzoin condensation ΔG: GFN2-xTB ohess + optional Shermo + ORCA SP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",       default="aldehydes_benzoin.csv")
    ap.add_argument("--smiles",      action="append", metavar="SMILES")
    ap.add_argument("--test",        action="store_true",
                    help="Run the 10 built-in test aldehydes")
    ap.add_argument("--output-dir",  default="thermo")
    ap.add_argument("--xtb-bin",     default=shutil.which("xtb") or "/home/schen3/xtb/bin/xtb")
    ap.add_argument("--shermo-bin",  default=_DEFAULT_SHERMO, metavar="PATH")
    ap.add_argument("--orca-bin",    default=_DEFAULT_ORCA,   metavar="PATH")
    ap.add_argument("--T",           type=float, default=298.15, metavar="K")
    ap.add_argument("--P",           type=float, default=1.0,    metavar="atm")
    ap.add_argument("--n-confs",     type=int,   default=10,     metavar="N")
    ap.add_argument("--conformer",   default="default", choices=["default", "funnel_v3"],
                    help="conformer ranker: 'funnel_v3' matches the xTB screen "
                         "(reactant+product consistent geometries); 'default' = ETKDG+xTB")
    ap.add_argument("--solvent",     default="dmso",
                    help="xTB ALPB + ORCA CPCM solvent (default: dmso; 'none' = gas phase)")
    ap.add_argument("--freq-cutoff", type=float, default=100.0,
                    help="Shermo qRRHO cutoff cm⁻¹ (default 100)")
    ap.add_argument("--sp-method",   default=None, metavar="FUNC",
                    help="ORCA DFT functional, e.g. PBE0-D4, B3LYP-D3BJ (omit to skip)")
    ap.add_argument("--sp-basis",    default="def2-TZVP", metavar="BASIS")
    ap.add_argument("--orca-nprocs", type=int, default=1, metavar="N",
                    help="MPI processes per ORCA call (default 1; requires OpenMPI in PATH)")
    ap.add_argument("--orca-maxcore",type=int, default=1792, metavar="MB",
                    help="Memory per ORCA process in MB (default 1792)")
    ap.add_argument("--max-atoms",   type=int, default=0, metavar="N",
                    help="Skip ORCA SP if heavy atoms > N (0 = no limit)")
    ap.add_argument("--ensemble-k",  type=int, default=1, metavar="K",
                    help="Boltzmann-average the DFT ΔG over the top-K xTB conformers "
                         "(1 = single best conformer; >1 fixes conformer noise for "
                         "flexible molecules)")
    ap.add_argument("--ensemble-window", type=float, default=3.0, metavar="KCAL",
                    help="Only include conformers within this xTB energy window (kcal/mol)")
    ap.add_argument("--skip",        type=int, default=0,
                    help="Skip first N records")
    ap.add_argument("--max",         type=int, default=0)
    ap.add_argument("--workers",     type=int, default=1)
    ap.add_argument("--timeout",     type=int, default=3600)
    ap.add_argument("--log-level",   default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.getLogger().setLevel(args.log_level)

    # ── Locate xtb ────────────────────────────────────────────────────────
    xtb_bin = shutil.which(args.xtb_bin) or args.xtb_bin
    try:
        r = subprocess.run([xtb_bin, "--version"], capture_output=True, text=True, timeout=10)
        ver = next((l.strip() for l in r.stdout.splitlines() if "xtb" in l.lower()), "")
        log.info("xtb: %s  (%s)", xtb_bin, ver)
    except Exception:
        log.error("xtb not found: %s", xtb_bin)
        return 1

    # ── Locate Shermo ─────────────────────────────────────────────────────
    shermo_bin = None
    if args.shermo_bin:
        p = shutil.which(args.shermo_bin) or args.shermo_bin
        if Path(p).exists():
            shermo_bin = p
            log.info("Shermo: %s", shermo_bin)
        else:
            log.warning("Shermo not found: %s  — Shermo disabled", p)

    # ── Locate ORCA ───────────────────────────────────────────────────────
    orca_bin = args.orca_bin
    if args.sp_method:
        if not Path(orca_bin).exists():
            log.error("ORCA not found: %s  — cannot run SP", orca_bin)
            return 1
        log.info("ORCA: %s  method=%s/%s  nprocs=%d  maxcore=%d MB",
                 orca_bin, args.sp_method, args.sp_basis,
                 args.orca_nprocs, args.orca_maxcore)

    solvent = "" if args.solvent.lower() == "none" else args.solvent
    if solvent:
        orca_solvent = _ORCA_SOLVENT.get(solvent.lower(), "")
        log.info("Solvation: ALPB(%s)/CPCM(%s)", solvent, orca_solvent or "—unknown—")
    else:
        log.info("Solvation: gas phase")

    # ── Build records ─────────────────────────────────────────────────────
    records: list[dict] = []
    if args.test:
        records = [{"aldehyde_name": n, "aldehyde_smiles": s} for n, s in _TEST_ALDEHYDES]
        log.info("Test mode: %d built-in aldehydes", len(records))
    elif args.smiles:
        records = [{"aldehyde_name": f"input_{i}", "aldehyde_smiles": s}
                   for i, s in enumerate(args.smiles)]
    else:
        csv_path = Path(args.input)
        if not csv_path.exists():
            log.error("Input not found: %s  (use --test or --smiles for quick tests)", csv_path)
            return 1
        with csv_path.open(encoding="utf-8") as fh:
            records = list(csv.DictReader(fh))
        log.info("Loaded %d records from %s", len(records), csv_path)

    if args.skip:
        records = records[args.skip:]
        log.info("Skipped first %d records", args.skip)
    if args.max:
        records = records[: args.max]
        log.info("Capped at %d", len(records))

    out_dir   = Path(args.output_dir)
    work_root = out_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    kw = dict(
        work_root=work_root, xtb_bin=xtb_bin, shermo_bin=shermo_bin,
        T=args.T, P_atm=args.P, n_confs_max=args.n_confs,
        solvent=solvent, freq_cutoff=args.freq_cutoff,
        sp_method=args.sp_method, sp_basis=args.sp_basis,
        orca_bin=orca_bin, orca_nprocs=args.orca_nprocs,
        orca_maxcore=args.orca_maxcore, max_atoms=args.max_atoms,
        ensemble_k=args.ensemble_k, ensemble_window=args.ensemble_window,
        conformer=args.conformer,
    )

    results: list[dict] = []
    if args.workers > 1:
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for idx, rec in enumerate(records):
                futures[ex.submit(calc_thermo_one, rec, idx, **kw)] = idx
        for fut in tqdm(as_completed(futures), total=len(futures), desc="thermo"):
            try:
                results.append(fut.result())
            except Exception as exc:
                log.error("Worker %d failed: %s", futures[fut], exc)
        results.sort(key=lambda r: r.get("idx") or 0)
    else:
        for idx, rec in enumerate(tqdm(records, desc="thermo")):
            results.append(calc_thermo_one(rec, idx, **kw))

    csv_out = out_dir / "delta_G.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    n_ok   = sum(1 for r in results if r.get("dG_xtb_kcal")        is not None)
    n_sh   = sum(1 for r in results if r.get("dG_shermo_kcal")     is not None)
    n_sp   = sum(1 for r in results if r.get("dG_orca_kcal")       is not None)
    n_spsh = sum(1 for r in results if r.get("dG_orca_shermo_kcal") is not None)
    n_err  = sum(1 for r in results if r.get("error", ""))

    log.info("══════════════════════════════════════════════════════")
    log.info("Total pairs                    : %d", len(results))
    log.info("ΔG(xTB+ALPB)                   : %d", n_ok)
    if shermo_bin:
        log.info("ΔG(Shermo+ALPB)                : %d", n_sh)
    if args.sp_method:
        log.info("ΔG(ORCA+xTBRRHO)               : %d", n_sp)
        log.info("ΔG(ORCA+ShermoRRHO)            : %d", n_spsh)
    log.info("Pairs with errors              : %d", n_err)
    log.info("Output CSV                     : %s", csv_out)

    plot_results(results, out_dir, shermo_bin, args.sp_method,
                 sp_basis=args.sp_basis, solvent=solvent)

    if n_ok:
        vals = [r["dG_xtb_kcal"] for r in results if r.get("dG_xtb_kcal") is not None]
        log.info("ΔG(xTB/%s) range  : %+.2f → %+.2f kcal/mol", solvent, min(vals), max(vals))
    if n_sp:
        vals_sp = [r["dG_orca_kcal"] for r in results if r.get("dG_orca_kcal") is not None]
        log.info("ΔG(%s/%s) range: %+.2f → %+.2f kcal/mol",
                 args.sp_method, args.sp_basis, min(vals_sp), max(vals_sp))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
