#!/usr/bin/env python
"""Rec-1 production-geometry recheck.

Question: is the D-pilot's low B97-3c residual scatter (std ratio 0.26) an artefact
of the pilot's ONE-SHOT ETKDG-seed42 -> MMFF -> GFN2 conformer, or does it survive on
the REAL production funnel_v3 + GFN2-ohess geometry protocol that the r2SCAN-3c labels
were actually computed on?

Per pilot holdout pair, for each of {product, donor aldehyde, acceptor aldehyde}:
  conf_funnel_v3 conformer rank  ->  GFN2 --ohess tight --alpb dmso     (== the production
  label geometry protocol, thermo_orca._mol_pipeline)  ->  THREE single points on that
  identical xtbopt.xyz geometry:
      E_r2scan  r2SCAN-3c / CPCM(DMSO)    -- the project label level
      E_b973c   B97-3c    / CPCM(DMSO)    -- the candidate cheaper baseline
      E_gxtb    g-xTB     / COSMO(DMSO)   -- the current production baseline level

  composite  G_lvl = E_lvl + (G_xtb_ohess - E_el_xtb_ohess)   [thermal identical across lvl]
  dG_lvl     = (G_lvl[prod] - G_lvl[don] - G_lvl[acc]) * HARTREE

Reported (merge script aggregates std):
  resid_gxtb  = dG_r2scan - dG_gxtb     within-geometry, thermal cancels  <- current model
  resid_b973c = dG_r2scan - dG_b973c    within-geometry, thermal cancels  <- B97-3c model
  repro_r2scan = dG_r2scan - label_stored   funnel_v3 label-reproduction (conformer noise)
  repro_gxtb   = dG_gxtb   - gxtb_stored    production g-xTB baseline reproduction

Usage (array, one pair per task):
  python rec1_prodgeom_recheck_worker.py --sample <cand.csv> --skip $i --max 1 \
      --out <chunk_$i.csv> --scratch $TMPDIR --xtb-cores 8
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, "/gpfs/scratch1/shared/schen3/benzoin-dg-restored/pipeline/compute")
import conf_funnel_v3  # noqa: E402,F401  (import FIRST -- breaks a thermo_orca<->conf_funnel circular import)
import thermo_orca as T  # noqa: E402

RDLogger.DisableLog("rdApp.*")
HK = 627.509474
XTB = os.environ.get("XTB_BIN", "/home/schen3/xtb/bin/xtb")
ORCA = os.environ.get("ORCA_BIN", "/home/schen3/orca/orca")
_GXTB_E = re.compile(r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh")


def _charge(smi: str) -> int:
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.GetFormalCharge(m) if m is not None else 0
    except Exception:
        return 0


def _gxtb_sp(geom: Path, wd: Path, chg: int, solv: str = "dmso") -> tuple[float | None, str]:
    """g-xTB COSMO(DMSO) SP (Eh) on `geom`, matching the production dG_gxtb baseline
    (cb_featurize._gxtb_sp). Falls back to gas phase and flags it if COSMO fails."""
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "m.xyz").write_text(Path(geom).read_text())
    for mode, extra in (("cosmo", ["--cosmo", solv]), ("gas", [])):
        r = subprocess.run([XTB, "m.xyz", "--gxtb", "--sp", "--chrg", str(chg), *extra],
                           cwd=wd, capture_output=True, text=True, timeout=1800)
        (wd / f"gxtb_{mode}.log").write_text(r.stdout + r.stderr)
        m = _GXTB_E.findall(r.stdout + r.stderr)
        if m:
            return float(m[-1]), mode
    return None, "fail"


def _species(role: str, smi: str, wd: Path, xtb_cores: int) -> dict:
    """funnel_v3 rank -> GFN2 ohess -> 3 SPs on xtbopt.xyz. Returns component energies (Eh)."""
    out = {"role": role, "err": None}
    ranked = conf_funnel_v3.rank_conformers_funnel_v3(
        smi, wd / "conf", XTB, n_confs_max=0, title=role, solvent="dmso", cores=xtb_cores, workers=1)
    if not ranked:
        out["err"] = f"{role}:funnel_embed_fail"
        return out
    best_xyz = ranked[0][0]
    stdout, _ = T.run_ohess(best_xyz, wd / "ohess", XTB, solvent="dmso", cores=xtb_cores, timeout=5400)
    G = T.parse_xtb_G(stdout)
    E_el = T._parse_xtb_energy(stdout)
    geom = wd / "ohess" / "xtbopt.xyz"
    if G is None or E_el is None or not geom.exists():
        out["err"] = f"{role}:ohess_fail(G={G is not None},E={E_el is not None},geom={geom.exists()})"
        return out
    out["G_xtb"], out["E_el_xtb"], out["thermal"] = G, E_el, G - E_el
    chg = _charge(smi)
    out["E_r2scan"] = T.calc_orca_sp(geom, "r2SCAN-3c", "", "DMSO", charge=chg,
                                     maxcore_mb=2500, orca_bin=ORCA, timeout=10800)
    out["E_b973c"] = T.calc_orca_sp(geom, "B97-3c", "", "DMSO", charge=chg,
                                    maxcore_mb=2500, orca_bin=ORCA, timeout=10800)
    out["E_gxtb"], out["gxtb_mode"] = _gxtb_sp(geom, wd / "gxtb", chg)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--xtb-cores", type=int, default=8)
    a = ap.parse_args()

    df = pd.read_csv(a.sample).iloc[a.skip:a.skip + a.max]
    wroot = Path(a.scratch) / f"rec1_{a.skip}"
    wroot.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in df.itertuples():
        rec = {"pid": r.pid, "grp": getattr(r, "grp", None),
               "label_stored": getattr(r, "label", None),
               "gxtb_stored": getattr(r, "gxtb_stored", None), "error": None}
        try:
            jobs = [("prod", r.prod_smiles), ("don", r.donor_smiles), ("acc", r.acceptor_smiles)]
            with ProcessPoolExecutor(max_workers=3) as ex:
                futs = {role: ex.submit(_species, role, smi, wroot / f"{r.Index}_{role}", a.xtb_cores)
                        for role, smi in jobs}
                sp = {role: f.result() for role, f in futs.items()}
            for role in ("prod", "don", "acc"):
                if sp[role]["err"]:
                    rec["error"] = sp[role]["err"]
                    break
            if rec["error"] is None:
                for lvl in ("r2scan", "b973c", "gxtb"):
                    ek = f"E_{lvl}"
                    if any(sp[x].get(ek) is None for x in ("prod", "don", "acc")):
                        rec["error"] = f"missing {ek}"
                        break
                    G = {x: sp[x][ek] + sp[x]["thermal"] for x in ("prod", "don", "acc")}
                    rec[f"dG_{lvl}_kcal"] = (G["prod"] - G["don"] - G["acc"]) * HK
                if rec["error"] is None:
                    rec["resid_gxtb"] = rec["dG_r2scan_kcal"] - rec["dG_gxtb_kcal"]
                    rec["resid_b973c"] = rec["dG_r2scan_kcal"] - rec["dG_b973c_kcal"]
                    if rec["label_stored"] is not None:
                        rec["repro_r2scan"] = rec["dG_r2scan_kcal"] - float(rec["label_stored"])
                    if rec["gxtb_stored"] is not None and pd.notna(rec["gxtb_stored"]):
                        rec["repro_gxtb"] = rec["dG_gxtb_kcal"] - float(rec["gxtb_stored"])
                    rec["gxtb_mode"] = ",".join(sorted({sp[x].get("gxtb_mode", "?") for x in ("prod", "don", "acc")}))
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[:180]
        print(f"  pid={rec['pid']} dG_r2scan={rec.get('dG_r2scan_kcal')} "
              f"resid_gxtb={rec.get('resid_gxtb')} resid_b973c={rec.get('resid_b973c')} "
              f"repro_r2scan={rec.get('repro_r2scan')} err={rec['error']}", flush=True)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    shutil.rmtree(wroot, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
