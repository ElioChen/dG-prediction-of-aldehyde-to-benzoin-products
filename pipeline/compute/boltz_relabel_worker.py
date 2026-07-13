#!/usr/bin/env python
"""Better-label probe: multi-conformer Boltzmann-averaged DFT + higher-level functional.

For each test molecule we re-derive the *electronic* reaction energy ΔE = E_prod - 2*E_ald
(homo-benzoin, 2 RCHO -> 1 product) three ways, on freshly generated conformers:

  dE_r2scan_single : lowest-xTB-conformer r2SCAN-3c (reproduces the current label's DFT part)
  dE_r2scan_boltz  : xTB-Boltzmann-averaged r2SCAN-3c over K conformers (the cleaner label)
  dE_wb97x_single  : lowest-xTB-conformer wB97X-3c   (range-separated hybrid; functional A/B)

The current label is dG = ΔE_elec(r2SCAN-3c, single-conf) + ΔG_thermal(xTB, per-mol). So the
*label correction* from each route is just the ΔE shift (thermal term unchanged):
  Boltzmann correction = dE_r2scan_boltz - dE_r2scan_single
  functional shift      = dE_wb97x_single - dE_r2scan_single
Their spread bounds how much room better labels have to move the ~1.6 kcal MAE floor.

Both species get K conformers (ETKDGv3 seed42 -> MMFF -> xTB-opt in DMSO -> ORCA SP),
Boltzmann weights from the xTB energies. Reuses thermo_orca (same level as the labels)."""
import argparse, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/scratch-shared/schen3/benzoin-dg/pipeline/compute")
import conf_funnel_v3  # noqa: F401  import BEFORE thermo_orca (breaks funnel<->thermo circular import)
import thermo_orca as T
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

XTB = os.environ.get("XTB_BIN", "/home/schen3/xtb/bin/xtb")
ORCA = "/home/schen3/orca/orca"
HK = 627.509474
KCAL_RT = 0.5925  # RT at 298 K, kcal/mol
K = int(os.environ.get("NCONF", "5"))


def confs_to_xyz(smi, k):
    m = Chem.AddHs(Chem.MolFromSmiles(str(smi)))
    if m is None:
        return []
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(m, numConfs=k, params=p)
    if not cids:
        return []
    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500)
    out = []
    for c in cids:
        conf = m.GetConformer(c)
        lines = [str(m.GetNumAtoms()), ""]
        for i, a in enumerate(m.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            lines.append(f"{a.GetSymbol()} {pos.x:.5f} {pos.y:.5f} {pos.z:.5f}")
        out.append("\n".join(lines) + "\n")
    return out


def species_energies(smi, wd):
    """Return dict with r2SCAN single/boltz (Eh), wB97X single (Eh), nconf, std(kcal)."""
    xyzs = confs_to_xyz(smi, K)
    Extb, Er2, geoms = [], [], []
    for ci, xyz in enumerate(xyzs):
        d = wd / f"c{ci}"; d.mkdir(parents=True, exist_ok=True)
        opt_xyz, ex = T._xtb_opt_energy(xyz, d, XTB, solvent="dmso")
        if opt_xyz is None or ex is None:
            continue
        (d / "opt.xyz").write_text(opt_xyz)
        e2 = T.calc_orca_sp(d / "opt.xyz", "r2SCAN-3c", "def2-mTZVP", "DMSO",
                            maxcore_mb=2000, orca_bin=ORCA, timeout=5400)
        if e2 is None:
            shutil.rmtree(d, ignore_errors=True); continue
        Extb.append(ex); Er2.append(e2); geoms.append(d / "opt.xyz")
    if not Er2:
        return None
    Extb = np.array(Extb); Er2 = np.array(Er2)
    imin = int(np.argmin(Extb))
    w = np.exp(-(Extb - Extb.min()) * HK / KCAL_RT); w /= w.sum()
    # higher functional only on the single lowest-xTB conformer (cheap A/B)
    ewb = T.calc_orca_sp(geoms[imin], "wB97X-3c", "", "DMSO",
                         maxcore_mb=2000, orca_bin=ORCA, timeout=5400)
    for g in geoms:
        shutil.rmtree(g.parent, ignore_errors=True)
    return dict(
        r2_single=float(Er2[imin]),
        r2_boltz=float((w * Er2).sum()),
        wb_single=(float(ewb) if ewb is not None else np.nan),
        nconf=len(Er2),
        r2_std_kcal=float((Er2 - Er2.min()).std() * HK),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    df = pd.read_csv(a.sample).iloc[a.skip:a.skip + a.max]
    root = Path(a.scratch) / f"br_{a.skip}"; root.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in df.itertuples():
        rec = {"id": r.id, "error": None}
        try:
            ald = species_energies(r.ald_smiles, root / f"a{r.id}")
            prod = species_energies(r.prod_smiles, root / f"p{r.id}")
            if ald and prod:
                rec.update(
                    dE_r2_single=(prod["r2_single"] - 2 * ald["r2_single"]) * HK,
                    dE_r2_boltz=(prod["r2_boltz"] - 2 * ald["r2_boltz"]) * HK,
                    dE_wb_single=(prod["wb_single"] - 2 * ald["wb_single"]) * HK,
                    nconf_ald=ald["nconf"], nconf_prod=prod["nconf"],
                    std_ald_kcal=ald["r2_std_kcal"], std_prod_kcal=prod["r2_std_kcal"],
                )
            else:
                rec["error"] = "ald or prod failed"
        except Exception as e:
            rec["error"] = str(e)[:90]
        rows.append(rec)
        print(f"  id={r.id} boltz_corr="
              f"{(rec.get('dE_r2_boltz', float('nan')) - rec.get('dE_r2_single', float('nan'))):.3f} "
              f"func_shift={(rec.get('dE_wb_single', float('nan')) - rec.get('dE_r2_single', float('nan'))):.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
