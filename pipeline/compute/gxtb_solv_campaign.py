#!/usr/bin/env python
"""Same-molecule, fully-solvated benzoin ΔG campaign on the 1% pilot — BOTH geometry routes.

For each aldehyde (+ its homo-benzoin product), all in DMSO continuum:

  Option A  — funnel_v3 (GFN2/CREST) geometry, then 3 single-points ON THAT geometry:
                GFN2  : --ohess tight --alpb dmso         -> G (ALPB)
                g-xTB : --gxtb --sp   --cosmo dmso         -> E (COSMO)
                DFT   : r2SCAN-3c SP  CPCM(DMSO)           -> E (CPCM)
  Option B  — g-xTB --gxtb --opt --cosmo dmso geometry (the "g-xTB optimises in solvent"
                variant the user asked for), then g-xTB-COSMO SP + DFT r2SCAN-3c CPCM SP
                on the g-xTB-optimised geometry.

Reaction ΔG = [val(product) − 2·val(aldehyde)] (+ GFN2 RRHO thermal applied identically to
every electronic energy so the comparison is geometry/method-only).

SOLVENT RISK & MITIGATIONS (see header notes in REPORT):
  * g-xTB solvated GEOMETRY OPT has an inconsistent gradient (Grimme caveat): --gbe unstable,
    --cosmo gradient lacks solute response, --alpb/--gbsa FATAL-error. We therefore (a) use
    --cosmo (least-bad, conductor-like ≈ CPCM), (b) record opt convergence + a coarse heavy-atom
    deviation vs the funnel geometry to flag blow-ups, and (c) treat Option B as EXPLORATORY —
    Option A (trusted GFN2/funnel geometry) is the reference comparison.
  * Continuum-model mismatch: DFT CPCM(DMSO) and g-xTB COSMO(DMSO) are both conductor-like
    (most comparable); GFN2 ALPB is a different (generalized-Born) family — flagged, not mixed.
  * NO GAS-PHASE FALLBACK: if any solvated step fails, that value is left blank (never
    back-filled with gas phase), so the dataset stays consistently solvated.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conf_funnel_v3            # noqa: E402  (import order matters; see gxtb_baseline.py)
import thermo_orca as Th         # noqa: E402

K = Th.HARTREE_TO_KCAL
XTB_GFN2 = os.environ.get("XTB_BIN", "/home/schen3/xtb/bin/xtb")
XTB_GXTB = os.environ.get(
    "GXTB_BIN", "/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb")
ORCA = os.environ.get("ORCA_BIN", "/home/schen3/orca/orca")
SOLVENT = "dmso"          # xTB ALPB / g-xTB COSMO solvent
ORCA_SOLV = "DMSO"        # ORCA CPCM solvent
METHOD = "r2SCAN-3c"
ORCA_NPROCS = int(os.environ.get("ORCA_NPROCS", "1"))     # 1=serial (no mpirun); >1 needs OpenMPI 4.1.6
ORCA_MAXCORE = int(os.environ.get("ORCA_MAXCORE", "3500"))
T, P_ATM = 298.15, 1.0
_GXTB_E = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)|energy\s+(-?\d+\.\d+)\s*Eh", re.I)
_GXTB_E2 = re.compile(r"(-?\d+\.\d+)\s+Eh")


def _charge(smiles: str) -> int:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        return Chem.GetFormalCharge(m) if m else 0
    except Exception:
        return 0


def _read_heavy(xyz: Path):
    """heavy-atom coords (centroid-removed) for a coarse deviation proxy."""
    lines = xyz.read_text().splitlines()
    pts = []
    for ln in lines[2:]:
        p = ln.split()
        if len(p) >= 4 and p[0].upper() != "H":
            try:
                pts.append((float(p[1]), float(p[2]), float(p[3])))
            except ValueError:
                pass
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    return [(x - cx, y - cy, z - cz) for x, y, z in pts]


def _coarse_rmsd(a: Path, b: Path):
    """centroid-removed, NON-aligned heavy-atom RMSD — only a blow-up flag, not a metric."""
    A, B = _read_heavy(a), _read_heavy(b)
    if not A or not B or len(A) != len(B):
        return ""
    s = sum((A[i][0]-B[i][0])**2 + (A[i][1]-B[i][1])**2 + (A[i][2]-B[i][2])**2
            for i in range(len(A)))
    return round(math.sqrt(s / len(A)), 3)


def _gxtb_sp(geom: Path, wd: Path, charge: int, timeout: int = 900):
    """g-xTB COSMO(DMSO) single point -> total energy (Eh) or None. SOLVENT ASSERTED."""
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "g.xyz").write_text(geom.read_text())
    cmd = [XTB_GXTB, "g.xyz", "--gxtb", "--sp", "--chrg", str(charge), "--cosmo", SOLVENT]
    E = None
    try:
        r = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
        if not ("fatal" in out.lower() or "abnormal" in out.lower()):
            m = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", out)
            E = float(m[-1]) if m else None
    except subprocess.TimeoutExpired:
        pass
    shutil.rmtree(wd, ignore_errors=True)   # real-time cleanup: keep only the parsed energy
    return E


def _gxtb_cosmo_opt(geom_str: str, wd: Path, charge: int, timeout: int = 3600):
    """g-xTB --opt --cosmo dmso. Returns (opt_xyz_path|None, converged_bool). EXPLORATORY."""
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "in.xyz").write_text(geom_str)
    cmd = [XTB_GXTB, "in.xyz", "--gxtb", "--opt", "tight",
           "--chrg", str(charge), "--cosmo", SOLVENT]
    try:
        r = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, False
    out = r.stdout + r.stderr
    (wd / "gxtb_opt.log").write_text(out)
    conv = "GEOMETRY OPTIMIZATION CONVERGED" in out
    of = wd / "xtbopt.xyz"
    return (of if of.exists() else None), conv


def _dft_sp(geom: Path, wd: Path, charge: int, timeout: int = 10800):
    """DFT r2SCAN-3c CPCM(DMSO) single point. E_total (Eh) or None. nprocs via ORCA_NPROCS env."""
    E = Th.calc_orca_sp(geom, METHOD, "", ORCA_SOLV, charge=charge,
                        nprocs=ORCA_NPROCS, maxcore_mb=ORCA_MAXCORE, orca_bin=ORCA, timeout=timeout)
    # real-time cleanup: calc_orca_sp leaves ~140 ORCA files (cpcm/integrals/property/out);
    # we only need the energy float -> delete the whole orca_sp dir now, not at molecule end.
    shutil.rmtree(geom.parent / "orca_sp", ignore_errors=True)
    return E


def species(smiles: str, wd: Path, label: str) -> dict:
    """All solvated energies for one species, both geometry routes."""
    wd.mkdir(parents=True, exist_ok=True)
    c = _charge(smiles)
    d = {"charge": c, "err": ""}
    # --- funnel_v3 geometry + GFN2 ohess (ALPB) ---
    ranked = conf_funnel_v3.rank_conformers_funnel_v3(
        smiles, wd / "rank", XTB_GFN2, n_confs_max=0, title=label,
        solvent=SOLVENT, cores=1, workers=1)
    if not ranked:
        d["err"] = f"{label}:embed_fail"; return d
    best = ranked[0][0]
    stdout, _ = Th.run_ohess(best, wd / "ohess", XTB_GFN2, T, P_ATM, solvent=SOLVENT)
    G = Th.parse_xtb_G(stdout); E_el = Th._parse_xtb_energy(stdout)
    geom_f = wd / "ohess" / "xtbopt.xyz"
    if G is None or E_el is None or not geom_f.exists():
        d["err"] = f"{label}:ohess_fail"; return d
    d["G_gfn2"] = G
    d["thermal"] = G - E_el                      # GFN2 RRHO thermal, applied to all routes
    shutil.rmtree(wd / "rank", ignore_errors=True)   # real-time cleanup: conformer-search temps
    # --- Option A single-points on funnel geometry ---
    d["E_gxtb_A"] = _gxtb_sp(geom_f, wd / "gxtbA", c)
    d["E_dft_A"] = _dft_sp(geom_f, wd / "dftA", c)
    # --- Option B: g-xTB cosmo opt, then SPs on that geometry ---
    geom_g, conv = _gxtb_cosmo_opt(geom_f.read_text(), wd / "gopt", c)
    d["gxtbopt_conv"] = conv
    if geom_g is not None:
        d["rmsd_fg"] = _coarse_rmsd(geom_f, geom_g)
        d["E_gxtb_B"] = _gxtb_sp(geom_g, wd / "gxtbB", c)
        d["E_dft_B"] = _dft_sp(geom_g, wd / "dftB", c)
    return d


def _dG(val_bz, val_ald, th_bz, th_ald):
    if val_bz is None or val_ald is None:
        return ""
    return round((val_bz - 2 * val_ald + (th_bz - 2 * th_ald)) * K, 4)


FIELDS = ["index", "aldehyde_smiles", "benzoin_smiles", "charge_ald", "charge_bz",
          "dG_gfn2_alpb_A", "dG_gxtb_cosmo_A", "dG_dft_cpcm_A",
          "dG_gxtb_cosmo_B", "dG_dft_cpcm_B",
          "gxtbopt_conv_ald", "gxtbopt_conv_bz", "rmsd_ald", "rmsd_bz",
          "G_ald_gfn2_Eh", "G_bz_gfn2_Eh",
          "E_ald_gxtb_A_Eh", "E_bz_gxtb_A_Eh", "E_ald_dft_A_Eh", "E_bz_dft_A_Eh",
          "E_ald_gxtb_B_Eh", "E_bz_gxtb_B_Eh", "E_ald_dft_B_Eh", "E_bz_dft_B_Eh",
          "thermal_ald_Eh", "thermal_bz_Eh", "note"]


def process(rec, i, scratch):
    idx = rec.get("index", rec.get("idx", str(i)))
    ald = (rec.get("aldehyde_smiles") or rec.get("SMILES") or "").strip()
    out = {k: "" for k in FIELDS}
    out["index"] = idx; out["aldehyde_smiles"] = ald
    bz = (rec.get("benzoin_smiles") or "").strip() or Th._make_benzoin_smiles(ald)
    out["benzoin_smiles"] = bz or ""
    if not bz:
        out["note"] = "no_benzoin"; return out
    wd = Path(scratch) / f"mol_{i:06d}"
    try:
        a = species(ald, wd / "ald", "ald")
        b = species(bz, wd / "bz", "bz")
    except Exception as e:
        out["note"] = f"err:{e}"[:90]; shutil.rmtree(wd, ignore_errors=True); return out
    errs = [d.get("err") for d in (a, b) if d.get("err")]
    out["charge_ald"] = a.get("charge", ""); out["charge_bz"] = b.get("charge", "")
    if "thermal" in a and "thermal" in b:
        tha, thb = a["thermal"], b["thermal"]
        out["thermal_ald_Eh"], out["thermal_bz_Eh"] = tha, thb
        out["G_ald_gfn2_Eh"], out["G_bz_gfn2_Eh"] = a.get("G_gfn2", ""), b.get("G_gfn2", "")
        # GFN2 ALPB ΔG (G already includes thermal)
        if "G_gfn2" in a and "G_gfn2" in b:
            out["dG_gfn2_alpb_A"] = round((b["G_gfn2"] - 2 * a["G_gfn2"]) * K, 4)
        # Option A
        out["E_ald_gxtb_A_Eh"], out["E_bz_gxtb_A_Eh"] = a.get("E_gxtb_A", ""), b.get("E_gxtb_A", "")
        out["E_ald_dft_A_Eh"], out["E_bz_dft_A_Eh"] = a.get("E_dft_A", ""), b.get("E_dft_A", "")
        out["dG_gxtb_cosmo_A"] = _dG(b.get("E_gxtb_A"), a.get("E_gxtb_A"), thb, tha)
        out["dG_dft_cpcm_A"] = _dG(b.get("E_dft_A"), a.get("E_dft_A"), thb, tha)
        # Option B
        out["gxtbopt_conv_ald"], out["gxtbopt_conv_bz"] = a.get("gxtbopt_conv", ""), b.get("gxtbopt_conv", "")
        out["rmsd_ald"], out["rmsd_bz"] = a.get("rmsd_fg", ""), b.get("rmsd_fg", "")
        out["E_ald_gxtb_B_Eh"], out["E_bz_gxtb_B_Eh"] = a.get("E_gxtb_B", ""), b.get("E_gxtb_B", "")
        out["E_ald_dft_B_Eh"], out["E_bz_dft_B_Eh"] = a.get("E_dft_B", ""), b.get("E_dft_B", "")
        out["dG_gxtb_cosmo_B"] = _dG(b.get("E_gxtb_B"), a.get("E_gxtb_B"), thb, tha)
        out["dG_dft_cpcm_B"] = _dG(b.get("E_dft_B"), a.get("E_dft_B"), thb, tha)
    out["note"] = "ok" if not errs else ";".join(errs)
    shutil.rmtree(wd, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/scratch-shared/schen3/tmp"))
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")),
                    help="molecule-level parallelism (each ORCA stays serial nprocs=1; needs NO MPI). "
                         "This is how the production DFT-SP got throughput: many 1-core molecules at once.")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.input)))
    if a.skip:
        rows = rows[a.skip:]
    if a.max:
        rows = rows[:a.max]
    scratch = Path(a.scratch) / f"solvcamp_{os.getpid()}"
    n = len(rows)
    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); f.flush()
        done = 0
        with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = {ex.submit(process, rec, a.skip + i, scratch): i for i, rec in enumerate(rows)}
            for fut in as_completed(futs):
                out = fut.result()
                w.writerow(out); f.flush()
                done += 1
                print(f"[{done}/{n}] idx={out['index']} A_dft={out['dG_dft_cpcm_A']} "
                      f"B_dft={out['dG_dft_cpcm_B']} note={out['note']}", flush=True)


if __name__ == "__main__":
    main()
