#!/usr/bin/env python
"""Tier B — full self-consistent B97-3c relabel campaign worker.

Adapted from rec1_prodgeom_recheck_worker.py. Per pair, for each of
{product, donor aldehyde, acceptor aldehyde}:
  conf_funnel_v3 rank -> GFN2 --ohess tight --alpb dmso   (production label geom protocol)
  -> THREE single points on that identical xtbopt.xyz geometry:
       E_r2scan  r2SCAN-3c / CPCM(DMSO)   -- NEW self-consistent project label
       E_b973c   B97-3c    / CPCM(DMSO)   -- NEW cheap baseline
       E_gxtb    g-xTB     / COSMO(DMSO)  -- same-geometry g-xTB baseline (cheap, kept for
                                             a self-consistent full-scale A/B; ~few % of cost)
  composite G_lvl = E_lvl + (G_xtb_ohess - E_el_xtb_ohess)
  dG_lvl = (G_lvl[prod] - G_lvl[don] - G_lvl[acc]) * HARTREE

Emitted per pair (one CSV row, appended immediately for crash-safe resume):
  pid, grp, new_scaffold_split, label_stored, gxtb_stored,
  dG_r2scan_kcal, dG_b973c_kcal, dG_gxtb_kcal,
  resid_gxtb (=dG_r2scan-dG_gxtb), resid_b973c (=dG_r2scan-dG_b973c),
  repro_r2scan (=dG_r2scan-label_stored), repro_gxtb (=dG_gxtb-gxtb_stored),
  gxtb_mode, error

Usage (array, one CHUNK of pairs per task; resumes from a partially written --out):
  python rec1_b973c_tierB_worker.py --sample <pairs.csv> --skip $START --nrows $CHUNK \
      --out <shard_$k.csv> --scratch $TMPDIR --xtb-cores 7
"""
from __future__ import annotations
import argparse
import csv
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

FIELDS = ["pid", "grp", "new_scaffold_split", "label_stored", "gxtb_stored",
          "dG_r2scan_kcal", "dG_b973c_kcal", "dG_gxtb_kcal",
          "resid_gxtb", "resid_b973c", "repro_r2scan", "repro_gxtb",
          "gxtb_mode", "error"]


def _charge(smi: str) -> int:
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.GetFormalCharge(m) if m is not None else 0
    except Exception:
        return 0


def _gxtb_sp(geom: Path, wd: Path, chg: int, solv: str = "dmso") -> tuple[float | None, str]:
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


def _one_pair(r, wroot: Path, xtb_cores: int) -> dict:
    rec = {k: None for k in FIELDS}
    rec["pid"] = r.pid
    rec["grp"] = getattr(r, "grp", None)
    rec["new_scaffold_split"] = getattr(r, "new_scaffold_split", None)
    rec["label_stored"] = getattr(r, "label", None)
    rec["gxtb_stored"] = getattr(r, "gxtb_stored", None)
    try:
        jobs = [("prod", r.prod_smiles), ("don", r.donor_smiles), ("acc", r.acceptor_smiles)]
        with ProcessPoolExecutor(max_workers=3) as ex:
            futs = {role: ex.submit(_species, role, smi, wroot / f"{r.pid}_{role}", xtb_cores)
                    for role, smi in jobs}
            sp = {role: f.result() for role, f in futs.items()}
        for role in ("prod", "don", "acc"):
            if sp[role]["err"]:
                rec["error"] = sp[role]["err"]
                return rec
        for lvl in ("r2scan", "b973c", "gxtb"):
            ek = f"E_{lvl}"
            if any(sp[x].get(ek) is None for x in ("prod", "don", "acc")):
                rec["error"] = f"missing {ek}"
                return rec
            G = {x: sp[x][ek] + sp[x]["thermal"] for x in ("prod", "don", "acc")}
            rec[f"dG_{lvl}_kcal"] = (G["prod"] - G["don"] - G["acc"]) * HK
        rec["resid_gxtb"] = rec["dG_r2scan_kcal"] - rec["dG_gxtb_kcal"]
        rec["resid_b973c"] = rec["dG_r2scan_kcal"] - rec["dG_b973c_kcal"]
        if rec["label_stored"] is not None and pd.notna(rec["label_stored"]):
            rec["repro_r2scan"] = rec["dG_r2scan_kcal"] - float(rec["label_stored"])
        if rec["gxtb_stored"] is not None and pd.notna(rec["gxtb_stored"]):
            rec["repro_gxtb"] = rec["dG_gxtb_kcal"] - float(rec["gxtb_stored"])
        rec["gxtb_mode"] = ",".join(sorted({sp[x].get("gxtb_mode", "?") for x in ("prod", "don", "acc")}))
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:180]
    return rec


def _done_pids(out_path: Path) -> set:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    try:
        d = pd.read_csv(out_path)
        return set(d["pid"].astype(str))
    except Exception:
        return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--nrows", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--xtb-cores", type=int, default=7)
    a = ap.parse_args()

    df = pd.read_csv(a.sample).iloc[a.skip:a.skip + a.nrows]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_pids(out_path)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    wroot = Path(a.scratch) / f"tierB_{a.skip}"
    wroot.mkdir(parents=True, exist_ok=True)

    n_total = len(df)
    n_skip = 0
    with out_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        if write_header:
            w.writeheader()
            fh.flush()
        for r in df.itertuples():
            if str(r.pid) in done:
                n_skip += 1
                continue
            rec = _one_pair(r, wroot, a.xtb_cores)
            w.writerow(rec)
            fh.flush()
            os.fsync(fh.fileno())
            print(f"  pid={rec['pid']} split={rec['new_scaffold_split']} "
                  f"dG_r2scan={rec.get('dG_r2scan_kcal')} dG_b973c={rec.get('dG_b973c_kcal')} "
                  f"resid_gxtb={rec.get('resid_gxtb')} resid_b973c={rec.get('resid_b973c')} "
                  f"repro_r2scan={rec.get('repro_r2scan')} err={rec['error']}", flush=True)
            shutil.rmtree(wroot / f"{r.pid}_prod", ignore_errors=True)
            shutil.rmtree(wroot / f"{r.pid}_don", ignore_errors=True)
            shutil.rmtree(wroot / f"{r.pid}_acc", ignore_errors=True)
    print(f"chunk skip={a.skip} nrows={n_total}: {n_skip} already done, {n_total - n_skip} computed this run", flush=True)
    shutil.rmtree(wroot, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
