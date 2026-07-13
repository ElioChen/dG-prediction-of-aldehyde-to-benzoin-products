#!/usr/bin/env python3
"""
g-xTB baseline ΔG for the benzoin Δ-learning A/B  (experiment, 2026-06-22).

Goal: replace the SEMIEMPIRICAL BASELINE of the Δ-model (GFN2 `dG_xtb_kcal`) with
g-xTB, evaluated on the SAME funnel_v3 (GFN2-ohess) geometry the DFT label sits on,
so the A/B isolates the energy-method swap (GFN2 -> g-xTB) from conformer choice.

Per species (aldehyde, benzoin product):
  1. funnel_v3 conformer search + GFN2 `--ohess tight --alpb dmso`  (reproduces the
     geometry AND GFN2 free energy behind `dG_xtb_kcal`; deterministic).
  2. g-xTB `--gxtb --sp --cosmo dmso` on the ohess geometry  ->  E_gxtb (SOLVATED).
  3. G_gxtb = E_gxtb + (G_gfn2 - E_gfn2)   (reuse the GFN2 RRHO thermal correction;
     thermal corrections are insensitive to the electronic method, and g-xTB Hessians
     are expensive/less stable — see memory gxtb-install).

dG_gxtb = G_gxtb(benzoin) - 2*G_gxtb(aldehyde);  dG_xtb (recomputed) is a determinism
sanity check against the parquet.

SOLVENT (all three methods are solvated, per user requirement): GFN2 geometry/thermal
use ALPB(DMSO); the DFT label uses CPCM(DMSO); g-xTB uses COSMO(DMSO) — its ALPB/GBSA
flags fatal-error, but `--cosmo dmso` (conductor-like continuum, closest to the DFT
CPCM) and `--gbe dmso` both run. COSMO chosen for apples-to-apples with the CPCM label.
If the solvated g-xTB SP fails on a molecule, dG_gxtb is left blank (NOT silently
back-filled with gas phase) so the baseline stays consistently solvated.

Usage:
  python gxtb_baseline.py --input sel.csv --out out.csv [--skip N --max M --workers W]
    input CSV columns: index, SMILES   (SMILES = aldehyde)
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for thermo_orca / conf_funnel_v3

# NB import conf_funnel_v3 BEFORE thermo_orca: the two are mutually circular
# (thermo_orca imports conf_funnel_v3 at top level; conf_funnel_v2 imports symbols
# from thermo_orca). Loading conf_funnel_v3 first fully initialises thermo_orca via
# its import chain; the reverse order raises ImportError on _mol_rotbonds.
import conf_funnel_v3
import thermo_orca as Th

HARTREE_TO_KCAL = Th.HARTREE_TO_KCAL
XTB_GFN2 = os.environ.get("XTB_BIN", "/home/schen3/xtb/bin/xtb")
XTB_GXTB = os.environ.get(
    "GXTB_BIN", "/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb")
SOLVENT = "dmso"
T = 298.15
P_ATM = 1.0
_GXTB_E = re.compile(r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh")


GXTB_SOLV = os.environ.get("GXTB_SOLV", "cosmo dmso").split()  # ALPB/GBSA fatal-error; COSMO ~ CPCM


def _gxtb_sp(geom: Path, wd: Path, charge: int = 0, timeout: int = 900) -> float | None:
    """g-xTB COSMO(DMSO) single point on `geom` -> total energy (Eh)."""
    wd.mkdir(parents=True, exist_ok=True)
    g = wd / "g.xyz"
    g.write_text(Path(geom).read_text())
    cmd = [XTB_GXTB, "g.xyz", "--gxtb", "--sp", "--chrg", str(charge)]
    if GXTB_SOLV:
        cmd += ["--" + GXTB_SOLV[0], *GXTB_SOLV[1:]]
    try:
        r = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout + r.stderr
    (wd / "gxtb_sp.log").write_text(out)
    m = _GXTB_E.findall(out)
    return float(m[-1]) if m else None


def _species(smiles: str, wd: Path, label: str) -> dict:
    """funnel_v3 + GFN2 ohess + g-xTB SP. Returns G/E (GFN2) and E_gxtb (Eh)."""
    wd.mkdir(parents=True, exist_ok=True)
    charge = 0
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is not None:
            charge = Chem.GetFormalCharge(m)
    except Exception:
        pass
    ranked = conf_funnel_v3.rank_conformers_funnel_v3(
        smiles, wd / "rank", XTB_GFN2, n_confs_max=0, title=label,
        solvent=SOLVENT, cores=1, workers=1)
    if not ranked:
        return {"err": f"{label}:embed_failed"}
    best_xyz = ranked[0][0]
    stdout, _g98 = Th.run_ohess(best_xyz, wd / "ohess", XTB_GFN2, T, P_ATM, solvent=SOLVENT)
    G = Th.parse_xtb_G(stdout)
    E_el = Th._parse_xtb_energy(stdout)
    if G is None or E_el is None:
        return {"err": f"{label}:ohess_parse_failed"}
    geom = wd / "ohess" / "xtbopt.xyz"
    if not geom.exists():
        return {"err": f"{label}:no_ohess_geom"}
    E_gxtb = _gxtb_sp(geom, wd / "gxtb", charge=charge)
    if E_gxtb is None:
        return {"err": f"{label}:gxtb_sp_failed", "G": G, "E_el": E_el}
    return {"G": G, "E_el": E_el, "E_gxtb": E_gxtb,
            "G_gxtb": E_gxtb + (G - E_el)}  # reuse GFN2 thermal correction


def process_mol(args_tuple):
    i, rec, scratch = args_tuple
    idx = rec.get("index", rec.get("idx", str(i)))
    ald = rec.get("SMILES") or rec.get("aldehyde_smiles") or ""
    wd = Path(scratch) / f"mol_{i:06d}"
    t0 = time.time()
    out = dict(index=idx, aldehyde_smiles=ald, dG_xtb_kcal="", dG_gxtb_kcal="",
               G_ald_xtb_Eh="", G_bz_xtb_Eh="", E_ald_gxtb_Eh="", E_bz_gxtb_Eh="",
               note="")
    bz = Th._make_benzoin_smiles(ald)
    if bz is None:
        out["note"] = "benzoin_gen_failed"
        return i, out, time.time() - t0
    a = _species(ald, wd / "ald", "ald")
    b = _species(bz, wd / "bz", "bz")
    errs = [d["err"] for d in (a, b) if "err" in d]
    if "G" in a and "G" in b:
        out["dG_xtb_kcal"] = round((b["G"] - 2 * a["G"]) * HARTREE_TO_KCAL, 4)
        out["G_ald_xtb_Eh"] = a["G"]
        out["G_bz_xtb_Eh"] = b["G"]
    if "G_gxtb" in a and "G_gxtb" in b:
        out["dG_gxtb_kcal"] = round((b["G_gxtb"] - 2 * a["G_gxtb"]) * HARTREE_TO_KCAL, 4)
        out["E_ald_gxtb_Eh"] = a["E_gxtb"]
        out["E_bz_gxtb_Eh"] = b["E_gxtb"]
    out["note"] = "ok" if not errs else ";".join(errs)
    # bound node-local inode use at scale
    try:
        import shutil
        shutil.rmtree(wd, ignore_errors=True)
    except Exception:
        pass
    return i, out, time.time() - t0


FIELDS = ["index", "aldehyde_smiles", "dG_xtb_kcal", "dG_gxtb_kcal",
          "G_ald_xtb_Eh", "G_bz_xtb_Eh", "E_ald_gxtb_Eh", "E_bz_gxtb_Eh", "note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/scratch-shared/schen3/tmp"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input)))
    if args.skip:
        rows = rows[args.skip:]
    if args.max:
        rows = rows[:args.max]
    base = Path(args.scratch) / f"gxtbbase_{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    tasks = [(i, r, str(base)) for i, r in enumerate(rows)]
    res: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_mol, t) for t in tasks]
        for k, fut in enumerate(as_completed(futs)):
            i, rec, dt = fut.result()
            res[i] = rec
            print(f"[{k+1}/{len(tasks)}] idx={rec['index']} dG_xtb={rec['dG_xtb_kcal']} "
                  f"dG_gxtb={rec['dG_gxtb_kcal']} {rec['note']} {dt:.0f}s", flush=True)
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows([res[j] for j in sorted(res)])
    print(f"\nwrote {args.out}: {len(res)} mols")


if __name__ == "__main__":
    main()
