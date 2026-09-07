#!/usr/bin/env python
"""Task D pilot: would a better-but-still-cheap single-point method as the
Delta-learning baseline lower the label-noise floor?

The DFT-arbitration work found the g-xTB-vs-r2SCAN-3c gap is dominated by the
single-point METHOD level (|Delta_SP| ~16 vs |Delta_geom| ~5 kcal), so the one
untested lever left for accuracy is a better cheap SP as the baseline that the
ML only has to correct. g-xTB (semiempirical) -> B97-3c (a GGA composite DFT,
~5-20x cheaper than the r2SCAN-3c label) is the natural candidate.

For each pilot pair, ONE fresh geometry per species (ETKDGv3 seed42 -> MMFF ->
GFN2 opt tight), then THREE single points on that identical geometry so the
comparison is conformer-noise-free:

  dG_gxtb_sp   = g-xTB SP           (xtb --gxtb),          (E_prod - E_don - E_acc)
  dG_b973c     = B97-3c/CPCM(DMSO)  (ORCA),                        ""
  dG_r2scan    = r2SCAN-3c/CPCM(DMSO) (ORCA, = the project label level)  ""

Reported downstream (merge_cheap_baseline_pilot.py):
  resid_gxtb  = dG_r2scan - dG_gxtb_sp     <- what the current Delta-model learns
  resid_b973c = dG_r2scan - dG_b973c       <- what a B97-3c-baseline Delta-model would
  if std(resid_b973c) << std(resid_gxtb) AND mean|.| smaller  -> a full B97-3c
  baseline recompute + retrain is worth it; else the lever is dead.

  <py> cross_benzoin/cheap_baseline_pilot_worker.py --sample <s.csv> \
       --skip 0 --max 1 --out <chunk.csv> --scratch /scratch-local/...
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, "/gpfs/scratch1/shared/schen3/benzoin-dg-restored/pipeline/compute")
import thermo_orca as T  # noqa: E402

RDLogger.DisableLog("rdApp.*")
HK = 627.5094740631
XTB = os.environ.get("XTB_BIN", "/home/schen3/xtb/bin/xtb")
ORCA = "/home/schen3/orca/orca"


def _seed_xyz(smiles: str) -> str | None:
    m = Chem.AddHs(Chem.MolFromSmiles(str(smiles)))
    if m is None:
        return None
    p = AllChem.ETKDGv3()
    p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    c = m.GetConformer()
    lines = [str(m.GetNumAtoms()), ""]
    for i, a in enumerate(m.GetAtoms()):
        q = c.GetAtomPosition(i)
        lines.append(f"{a.GetSymbol()} {q.x:.5f} {q.y:.5f} {q.z:.5f}")
    return "\n".join(lines) + "\n"


def _gfn2_opt(start_xyz: str, wd: Path) -> str | None:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "in.xyz").write_text(start_xyz)
    r = subprocess.run([XTB, "in.xyz", "--gfn", "2", "--opt", "tight", "--chrg", "0"],
                       cwd=wd, capture_output=True, text=True, timeout=3600)
    (wd / "opt.log").write_text(r.stdout + r.stderr)
    p = wd / "xtbopt.xyz"
    return p.read_text() if p.exists() else None


_GXTB_E = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)")


def _gxtb_sp(xyz_str: str, wd: Path) -> float | None:
    """g-xTB single point (Hartree). Gas phase -- g-xTB implicit solvation is not
    generally available; the project's own dG_gxtb baseline is likewise gas-phase
    g-xTB, so this is the like-for-like comparison."""
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "m.xyz").write_text(xyz_str)
    r = subprocess.run([XTB, "m.xyz", "--gxtb", "--sp", "--chrg", "0"],
                       cwd=wd, capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    (wd / "gxtb.log").write_text(out)
    m = _GXTB_E.findall(out)
    return float(m[-1]) if m else None


def _orca_sp(xyz_str: str, wd: Path, method: str) -> float | None:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "m.xyz").write_text(xyz_str)
    return T.calc_orca_sp(wd / "m.xyz", method, "", "DMSO",
                          maxcore_mb=2500, orca_bin=ORCA, timeout=7200)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    df = pd.read_csv(a.sample).iloc[a.skip:a.skip + a.max]
    wroot = Path(a.scratch) / f"cbp_{a.skip}"
    wroot.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in df.itertuples():
        rec = {"id": r.id, "grp": getattr(r, "grp", None),
               "dG_orca_kcal_stored": getattr(r, "dG_orca_kcal", None),
               "dG_gxtb_kcal_stored": getattr(r, "dG_gxtb_kcal", None), "error": None}
        try:
            E = {"gxtb": {}, "b973c": {}, "r2scan": {}}
            ok = True
            for role, smi in (("prod", r.smiles), ("don", r.donor_smiles), ("acc", r.acceptor_smiles)):
                seed = _seed_xyz(smi)
                if seed is None:
                    rec["error"] = f"embed fail {role}"; ok = False; break
                wdr = wroot / f"{r.Index}_{role}"
                geom = _gfn2_opt(seed, wdr / "opt")
                if geom is None:
                    rec["error"] = f"gfn2 opt fail {role}"; ok = False; break
                E["gxtb"][role] = _gxtb_sp(geom, wdr / "sp_gxtb")
                E["b973c"][role] = _orca_sp(geom, wdr / "sp_b973c", "B97-3c")
                E["r2scan"][role] = _orca_sp(geom, wdr / "sp_r2scan", "r2SCAN-3c")
            if ok and all(E[l][x] is not None for l in E for x in ("prod", "don", "acc")):
                for lvl in E:
                    rec[f"dG_{lvl}_kcal"] = (E[lvl]["prod"] - E[lvl]["don"] - E[lvl]["acc"]) * HK
                rec["resid_gxtb"] = rec["dG_r2scan_kcal"] - rec["dG_gxtb_kcal"]
                rec["resid_b973c"] = rec["dG_r2scan_kcal"] - rec["dG_b973c_kcal"]
        except Exception as e:
            rec["error"] = str(e)[:150]
        print(f"  id={r.id} dG_r2scan={rec.get('dG_r2scan_kcal')} "
              f"resid_gxtb={rec.get('resid_gxtb')} resid_b973c={rec.get('resid_b973c')} "
              f"err={rec['error']}", flush=True)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    shutil.rmtree(wroot, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
