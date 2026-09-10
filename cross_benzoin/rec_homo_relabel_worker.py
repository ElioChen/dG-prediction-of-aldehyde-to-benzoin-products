#!/usr/bin/env python
"""HOMO full-library self-consistent B97-3c relabel campaign worker.

Homo analogue of rec1_b973c_tierB_worker.py. For a homo (self-condensation)
pair donor==acceptor, ONLY TWO distinct species matter:
  {product, aldehyde}
so this computes 2 species per pair instead of Tier B's 3 (the aldehyde is
not evaluated twice) -- ~1/3 less compute over ~160k pairs.

Per species: conf_funnel_v3 rank -> GFN2 --ohess tight --alpb dmso, then three
single points on that identical xtbopt.xyz geometry (r2SCAN-3c / B97-3c / g-xTB,
all CPCM or COSMO DMSO) -- reuses rec1_b973c_tierB_worker._species verbatim.

  homo dG_lvl = (G_lvl[prod] - 2 * G_lvl[ald]) * HARTREE

Emitted per pair (appended immediately, crash-safe resume by pid):
  pid, grp, new_scaffold_split, label_stored, gxtb_stored,
  dG_r2scan_kcal, dG_b973c_kcal, dG_gxtb_kcal,
  resid_gxtb, resid_b973c, repro_r2scan, repro_gxtb, gxtb_mode, error

Usage (array, one CHUNK per task, resumes from a partially written --out):
  python rec_homo_relabel_worker.py --sample <pairs.csv> --skip $START --nrows $CHUNK \
      --out <shard_$k.csv> --scratch $TMPDIR --xtb-cores 7
"""
from __future__ import annotations
import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/gpfs/scratch1/shared/schen3/benzoin-dg-restored/cross_benzoin")
from rec1_b973c_tierB_worker import FIELDS, HK, _species, _done_pids  # noqa: E402


def _one_pair(r, wroot: Path, xtb_cores: int) -> dict:
    rec = {k: None for k in FIELDS}
    rec["pid"] = r.pid
    rec["grp"] = getattr(r, "grp", None)
    rec["new_scaffold_split"] = getattr(r, "new_scaffold_split", None)
    rec["label_stored"] = getattr(r, "label", None)
    rec["gxtb_stored"] = getattr(r, "gxtb_stored", None)
    try:
        jobs = [("prod", r.prod_smiles), ("ald", r.donor_smiles)]
        with ProcessPoolExecutor(max_workers=2) as ex:
            futs = {role: ex.submit(_species, role, smi, wroot / f"{r.pid}_{role}", xtb_cores)
                    for role, smi in jobs}
            sp = {role: f.result() for role, f in futs.items()}
        for role in ("prod", "ald"):
            if sp[role]["err"]:
                rec["error"] = sp[role]["err"]
                return rec
        for lvl in ("r2scan", "b973c", "gxtb"):
            ek = f"E_{lvl}"
            if any(sp[x].get(ek) is None for x in ("prod", "ald")):
                rec["error"] = f"missing {ek}"
                return rec
            G = {x: sp[x][ek] + sp[x]["thermal"] for x in ("prod", "ald")}
            rec[f"dG_{lvl}_kcal"] = (G["prod"] - 2.0 * G["ald"]) * HK
        rec["resid_gxtb"] = rec["dG_r2scan_kcal"] - rec["dG_gxtb_kcal"]
        rec["resid_b973c"] = rec["dG_r2scan_kcal"] - rec["dG_b973c_kcal"]
        if rec["label_stored"] is not None and pd.notna(rec["label_stored"]):
            rec["repro_r2scan"] = rec["dG_r2scan_kcal"] - float(rec["label_stored"])
        if rec["gxtb_stored"] is not None and pd.notna(rec["gxtb_stored"]):
            rec["repro_gxtb"] = rec["dG_gxtb_kcal"] - float(rec["gxtb_stored"])
        rec["gxtb_mode"] = ",".join(sorted({sp[x].get("gxtb_mode", "?") for x in ("prod", "ald")}))
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:180]
    return rec


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
    wroot = Path(a.scratch) / f"homorelabel_{a.skip}"
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
                  f"resid_b973c={rec.get('resid_b973c')} repro_r2scan={rec.get('repro_r2scan')} "
                  f"err={rec['error']}", flush=True)
            shutil.rmtree(wroot / f"{r.pid}_prod", ignore_errors=True)
            shutil.rmtree(wroot / f"{r.pid}_ald", ignore_errors=True)
    print(f"chunk skip={a.skip} nrows={n_total}: {n_skip} already done, "
          f"{n_total - n_skip} computed this run", flush=True)
    shutil.rmtree(wroot, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
