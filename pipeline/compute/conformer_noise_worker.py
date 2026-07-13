#!/usr/bin/env python
"""Quantify the single-conformer LABEL-NOISE floor: for each product, generate K conformers,
xTB-opt + r2SCAN-3c SP each, and measure the spread of the DFT energy (= ΔG spread, since only
the product conformer varies). Reports per-molecule std + |lowest-xtb-conf − Boltzmann| in kcal.
Reuses thermo_orca._xtb_opt_energy + calc_orca_sp (same level as the labels)."""
import argparse, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/scratch-shared/schen3/benzoin-dg/pipeline/compute")
import conf_funnel_v3  # noqa: F401  (import BEFORE thermo_orca to break the funnel↔thermo circular import)
import thermo_orca as T
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
XTB = os.environ.get("XTB_BIN", "/home/schen3/xtb/bin/xtb")
ORCA = "/home/schen3/orca/orca"; HK = 627.509474; KCAL_RT = 0.5925  # RT at 298K in kcal
K = int(os.environ.get("NCONF", "5"))

def confs_to_xyz(smi, k):
    m = Chem.AddHs(Chem.MolFromSmiles(str(smi)))
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(m, numConfs=k, params=p)
    if not cids: return []
    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500)
    out = []
    for c in cids:
        conf = m.GetConformer(c); syms = [a.GetSymbol() for a in m.GetAtoms()]
        lines = [str(m.GetNumAtoms()), ""]
        for i, s in enumerate(syms):
            pos = conf.GetAtomPosition(i); lines.append(f"{s} {pos.x:.5f} {pos.y:.5f} {pos.z:.5f}")
        out.append("\n".join(lines) + "\n")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True); ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=2); ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    df = pd.read_csv(a.sample).iloc[a.skip:a.skip + a.max]
    wroot = Path(a.scratch) / f"cn_{a.skip}"; wroot.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in df.itertuples():
        rec = {"id": r.id, "nconf_ok": 0, "error": None}
        try:
            xyzs = confs_to_xyz(r.smiles, K)
            Extb, Eorca = [], []
            for ci, xyz in enumerate(xyzs):
                wd = wroot / f"m{r.id}_c{ci}"; wd.mkdir(parents=True, exist_ok=True)
                opt_xyz, ex = T._xtb_opt_energy(xyz, wd, XTB, solvent="dmso")
                if opt_xyz is None: continue
                (wd / "opt.xyz").write_text(opt_xyz)
                eo = T.calc_orca_sp(wd / "opt.xyz", "r2SCAN-3c", "def2-mTZVP", "DMSO",
                                    maxcore_mb=2000, orca_bin=ORCA, timeout=5400)
                if eo is not None and ex is not None:
                    Extb.append(ex); Eorca.append(eo)
                shutil.rmtree(wd, ignore_errors=True)
            n = len(Eorca)
            rec["nconf_ok"] = n
            if n >= 2:
                Eo = np.array(Eorca) * HK; Ex = np.array(Extb) * HK  # kcal
                Eo -= Eo.min(); Ex -= Ex.min()                       # relative
                w = np.exp(-Ex / KCAL_RT); w /= w.sum()
                rec["dG_std_kcal"] = float(Eo.std())
                rec["dG_range_kcal"] = float(Eo.max() - Eo.min())
                rec["single_minus_boltz_kcal"] = float(Eo[np.argmin(Ex)] - (w * Eo).sum())
        except Exception as e:
            rec["error"] = str(e)[:80]
        rows.append(rec); print(f"  id={r.id} nconf={rec['nconf_ok']} std={rec.get('dG_std_kcal')}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    shutil.rmtree(wroot, ignore_errors=True)

if __name__ == "__main__":
    main()
