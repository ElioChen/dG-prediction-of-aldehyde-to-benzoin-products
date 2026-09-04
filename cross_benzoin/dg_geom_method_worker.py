#!/usr/bin/env python
"""Does GFN2- vs g-xTB-optimised geometry change the r2SCAN-3c ΔG label (all 3 species)?

geom_bias_worker measured only E(product). This does the full three-species ΔΔG:

  for each of product / donor / acceptor:
      ETKDGv3(seed42) -> MMFF -> SAME start structure ->
        GFN2 opt  (xtb --gfn 2 --opt) -> r2SCAN-3c SP  = E_gfn2
        g-xTB opt (xtb --gxtb  --opt) -> r2SCAN-3c SP  = E_gxtb
  dG_gfn2 = (E_prod_gfn2 - E_don_gfn2 - E_acc_gfn2) * HARTREE
  dG_gxtb = (E_prod_gxtb - E_don_gxtb - E_acc_gxtb) * HARTREE
  ddG = dG_gxtb - dG_gfn2   <-- the label bias from the geometry method, AFTER the
       donor/acceptor cancellation. Also RMSD(GFN2 opt, g-xtb opt) per species so we
       know the structure actually moved (not just an energy artefact).

Holds the conformer fixed (same ETKDG start) so this isolates the geometry-method
effect from conformer noise. Thermal term is unchanged -> ddG here == d(label).

  <py> cross_benzoin/dg_geom_method_worker.py --sample <sample.csv> \
       --skip 0 --max 1 --out <chunk.csv> --scratch /scratch-local/...
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "/gpfs/scratch1/shared/schen3/benzoin-dg-restored/pipeline/compute")
import conf_funnel_v3  # noqa: F401
import thermo_orca as T
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

HK = 627.5094740631
XTB = os.environ.get("XTB_BIN", "/home/schen3/xtb/bin/xtb")
ORCA = "/home/schen3/orca/orca"


def _seed_xyz(smiles: str) -> str | None:
    m = Chem.AddHs(Chem.MolFromSmiles(str(smiles)))
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    c = m.GetConformer()
    lines = [str(m.GetNumAtoms()), ""]
    for i, a in enumerate(m.GetAtoms()):
        q = c.GetAtomPosition(i)
        lines.append(f"{a.GetSymbol()} {q.x:.5f} {q.y:.5f} {q.z:.5f}")
    return "\n".join(lines) + "\n"


def _opt(seed_xyz: str, wd: Path, flavour: str) -> str | None:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "in.xyz").write_text(seed_xyz)
    flag = ["--gxtb", "--opt", "tight"] if flavour == "gxtb" else ["--gfn", "2", "--opt", "tight"]
    r = subprocess.run([XTB, "in.xyz", *flag, "--chrg", "0"], cwd=wd,
                       capture_output=True, text=True, timeout=3600)
    (wd / f"{flavour}.log").write_text(r.stdout + r.stderr)
    p = wd / "xtbopt.xyz"
    return p.read_text() if p.exists() else None


def _sp(xyz_str: str, wd: Path) -> float | None:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "m.xyz").write_text(xyz_str)
    return T.calc_orca_sp(wd / "m.xyz", "r2SCAN-3c", "def2-mTZVP", "DMSO",
                          maxcore_mb=2000, orca_bin=ORCA, timeout=5400)


def _rmsd(xa: str, xb: str) -> float | None:
    try:
        pa = np.array([[float(v) for v in l.split()[1:4]] for l in xa.splitlines()[2:] if l.split()])
        pb = np.array([[float(v) for v in l.split()[1:4]] for l in xb.splitlines()[2:] if l.split()])
        if pa.shape != pb.shape:
            return None
        pa -= pa.mean(0); pb -= pb.mean(0)
        H = pa.T @ pb
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        return float(np.sqrt(((pa @ R.T - pb) ** 2).sum(1).mean()))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    df = pd.read_csv(a.sample).iloc[a.skip:a.skip + a.max]
    wroot = Path(a.scratch) / f"dgm_{a.skip}"; wroot.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in df.itertuples():
        rec = {"id": r.id, "grp": getattr(r, "grp", None), "dG_orca_kcal": getattr(r, "dG_orca_kcal", None),
               "error": None}
        try:
            E = {"gfn2": {}, "gxtb": {}}
            rms = {}
            ok = True
            for role, smi in (("prod", r.smiles), ("don", r.donor_smiles), ("acc", r.acceptor_smiles)):
                seed = _seed_xyz(smi)
                if seed is None:
                    rec["error"] = f"embed fail {role}"; ok = False; break
                geoms = {}
                for fl in ("gfn2", "gxtb"):
                    wd = wroot / f"{r.Index}_{role}_{fl}"
                    g = _opt(seed, wd / "opt", fl)
                    if g is None:
                        rec["error"] = f"{fl} opt fail {role}"; ok = False; break
                    geoms[fl] = g
                    E[fl][role] = _sp(g, wd / "sp")
                if not ok:
                    break
                rms[role] = _rmsd(geoms["gfn2"], geoms["gxtb"])
            if ok and all(E[fl][x] is not None for fl in E for x in ("prod", "don", "acc")):
                dg = {fl: (E[fl]["prod"] - E[fl]["don"] - E[fl]["acc"]) * HK for fl in E}
                rec["dG_gfn2geom_kcal"] = dg["gfn2"]
                rec["dG_gxtbgeom_kcal"] = dg["gxtb"]
                rec["ddG_kcal"] = dg["gxtb"] - dg["gfn2"]
                rec["rmsd_prod"] = rms.get("prod"); rec["rmsd_don"] = rms.get("don"); rec["rmsd_acc"] = rms.get("acc")
        except Exception as e:
            rec["error"] = str(e)[:120]
        print(f"  id={r.id} ddG={rec.get('ddG_kcal')} rmsd_prod={rec.get('rmsd_prod')}", flush=True)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    shutil.rmtree(wroot, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
