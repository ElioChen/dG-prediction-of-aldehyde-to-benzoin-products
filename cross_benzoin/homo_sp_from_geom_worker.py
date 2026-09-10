#!/usr/bin/env python
"""FAST homo relabel worker: DFT single-points on ARCHIVED geometries.

Per homo pair (manifest row): extract the product + aldehyde xyz from their
geom.tar.zst archives, run one ORCA SP per (species x method), reuse the
stored xTB RRHO thermal:
    G_lvl(species) = E_lvl_SP(species) + thermal(species)
    dG_lvl = (G_lvl[prod] - 2 * G_lvl[ald]) * 627.5094740631      # homo

No conformer search, no Hessian -- ~10-20x cheaper than the self-consistent
worker. Geometry-consistent with the g-xTB / descriptor library (same 2026-09
GFN2-opt geoms). Default methods: r2SCAN-3c (label) + B97-3c (delta baseline).

Emitted per pair (append + fsync, resume by id):
  id, donor_id, new_scaffold_split, label_stored,
  dG_r2scan_kcal, dG_b973c_kcal, E_prod_r2scan, E_ald_r2scan,
  E_prod_b973c, E_ald_b973c, repro_r2scan (=dG_r2scan-label_stored), error

Usage (array, one CHUNK per task, resume-safe):
  python homo_sp_from_geom_worker.py --manifest <m.csv> --skip $S --nrows $N \
      --out <shard_$k.csv> --scratch $TMPDIR --sp-workers 16
"""
from __future__ import annotations
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/gpfs/scratch1/shared/schen3/benzoin-dg-restored/pipeline/compute")
import conf_funnel_v3  # noqa: E402,F401  (precede thermo_orca -> break circular import)
import thermo_orca as T  # noqa: E402

HK = 627.5094740631
ORCA = os.environ.get("ORCA_BIN", "/home/schen3/orca/orca")
DMSO = T._ORCA_SOLVENT.get("dmso", "DMSO")
METHOD_KEY = {"r2SCAN-3c": "r2scan", "B97-3c": "b973c"}

FIELDS = ["id", "donor_id", "new_scaffold_split", "label_stored",
          "dG_r2scan_kcal", "dG_b973c_kcal",
          "E_prod_r2scan", "E_ald_r2scan", "E_prod_b973c", "E_ald_b973c",
          "repro_r2scan", "error"]


def _extract(arc: str, member: str, dst: Path) -> Path | None:
    out = dst / member.replace("/", "_")
    if out.exists():
        return out
    try:
        r = subprocess.run(["tar", "--zstd", "-xf", arc, "-C", str(dst), member],
                           capture_output=True, text=True, timeout=300)
        src = dst / member
        if r.returncode == 0 and src.exists():
            src.rename(out)
            return out
    except Exception:
        pass
    return None


def _sp(args):
    """Run one ORCA SP in its OWN scratch dir (calc_orca_sp writes orca_sp/ next
    to the xyz, so concurrent SPs on the same geometry MUST get separate copies)."""
    xyz, method, basis, charge, maxcore = args
    wd = Path(tempfile.mkdtemp(prefix="hsp_", dir=os.environ.get("TMPDIR", "/tmp")))
    try:
        local = wd / "mol.xyz"
        shutil.copy(xyz, local)
        return T.calc_orca_sp(local, method, basis, DMSO, charge=int(charge),
                              maxcore_mb=maxcore, orca_bin=ORCA, timeout=7200)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--nrows", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--methods", default="r2SCAN-3c,B97-3c")
    ap.add_argument("--sp-workers", type=int, default=16)
    ap.add_argument("--maxcore", type=int, default=1500)
    a = ap.parse_args()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]

    df = pd.read_csv(a.manifest).iloc[a.skip:a.skip + a.nrows].reset_index(drop=True)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            done = set(pd.read_csv(out_path)["id"].astype(str))
        except Exception:
            pass

    wroot = Path(a.scratch) / f"homosp_{a.skip}"
    geodir = wroot / "geoms"
    geodir.mkdir(parents=True, exist_ok=True)

    todo = [r for r in df.itertuples() if str(r.id) not in done]
    print(f"chunk skip={a.skip}: {len(df)} rows, {len(df)-len(todo)} already done, "
          f"{len(todo)} to do, methods={methods}", flush=True)

    # 1. extract every needed geom once
    geom = {}
    for r in todo:
        geom[("p", r.id)] = _extract(r.prod_arc, r.prod_mem, geodir)
        geom[("a", r.id)] = _extract(r.ald_arc, r.ald_mem, geodir)

    # 2. flatten to SP jobs, run pool
    jobs, jkey = [], []
    for r in todo:
        gp, ga = geom[("p", r.id)], geom[("a", r.id)]
        for meth in methods:
            for role, g, chg in (("prod", gp, r.charge_prod), ("ald", ga, r.charge_ald)):
                if g is None:
                    continue
                jobs.append((str(g), meth, "", chg, a.maxcore))
                jkey.append((r.id, meth, role))
    E = {}
    if jobs:
        with ProcessPoolExecutor(max_workers=a.sp_workers) as ex:
            for k, val in zip(jkey, ex.map(_sp, jobs)):
                E[k] = val

    # 3. assemble rows
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        if write_header:
            w.writeheader(); fh.flush()
        for r in todo:
            rec = {k: None for k in FIELDS}
            rec.update(id=r.id, donor_id=r.donor_id,
                       new_scaffold_split=getattr(r, "new_scaffold_split", None),
                       label_stored=getattr(r, "label_stored", None))
            gp, ga = geom[("p", r.id)], geom[("a", r.id)]
            if gp is None or ga is None:
                rec["error"] = f"geom_extract_fail(p={gp is not None},a={ga is not None})"
                w.writerow(rec); fh.flush(); os.fsync(fh.fileno()); continue
            try:
                for meth in methods:
                    mk = METHOD_KEY[meth]
                    ep, ea = E.get((r.id, meth, "prod")), E.get((r.id, meth, "ald"))
                    rec[f"E_prod_{mk}"], rec[f"E_ald_{mk}"] = ep, ea
                    if ep is None or ea is None:
                        rec["error"] = f"{meth}:sp_fail(p={ep is not None},a={ea is not None})"
                        break
                    G_prod = ep + float(r.thermal_prod)
                    G_ald = ea + float(r.thermal_ald)
                    rec[f"dG_{mk}_kcal"] = (G_prod - 2.0 * G_ald) * HK
                if rec["error"] is None and rec["dG_r2scan_kcal"] is not None \
                        and pd.notna(getattr(r, "label_stored", None)):
                    rec["repro_r2scan"] = rec["dG_r2scan_kcal"] - float(r.label_stored)
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)[:180]
            w.writerow(rec); fh.flush(); os.fsync(fh.fileno())
            print(f"  id={rec['id']} split={rec['new_scaffold_split']} "
                  f"dG_r2scan={rec['dG_r2scan_kcal']} dG_b973c={rec['dG_b973c_kcal']} "
                  f"repro={rec['repro_r2scan']} err={rec['error']}", flush=True)
    shutil.rmtree(wroot, ignore_errors=True)
    print(f"chunk skip={a.skip} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
