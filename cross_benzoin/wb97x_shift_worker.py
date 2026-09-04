#!/usr/bin/env python
"""Functional-shift probe: r2SCAN-3c vs wB97X-3c single-point on the SAME geometry.

The cross-benzoin ΔG labels use r2SCAN-3c SP. If a range-separated hybrid
(wB97X-3c) systematically disagrees on E(product) by more than the ~2.2 kcal/mol
model MAE, then r2SCAN-3c carries a functional-level bias and better labels (a
higher functional) could move the floor. Per-species (product) E shift; the
label's ΔG = ΔE_elec + xTB thermal, so a per-species E shift maps ~directly to a
label shift.

  <feat-py> cross_benzoin/wb97x_shift_worker.py --sample <sample.csv> \
      --geom-dir <geoms> --skip 0 --max 2 --out <chunk.csv> --scratch /scratch-local/...
"""
from __future__ import annotations
import argparse, os, shutil, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "/gpfs/scratch1/shared/schen3/benzoin-dg-restored/pipeline/compute")
import conf_funnel_v3  # noqa: F401
import thermo_orca as T

HK = 627.5094740631
ORCA = "/home/schen3/orca/orca"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--geom-dir", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()

    df = pd.read_csv(a.sample).iloc[a.skip:a.skip + a.max]
    gd = Path(a.geom_dir)
    wroot = Path(a.scratch) / f"wb_{a.skip}"; wroot.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in df.itertuples():
        rec = {"id": r.id, "smiles": getattr(r, "smiles", None), "error": None}
        try:
            src = gd / f"prod_{r.id}.xyz"
            if not src.exists():
                rec["error"] = "no geom"; rows.append(rec); continue
            wd = wroot / f"m_{abs(hash(r.id)) % 10**8}"; wd.mkdir(parents=True, exist_ok=True)
            xyz = wd / "mol.xyz"; xyz.write_text(src.read_text())
            e_r2 = T.calc_orca_sp(xyz, "r2SCAN-3c", "def2-mTZVP", "DMSO",
                                  maxcore_mb=2000, orca_bin=ORCA, timeout=5400)
            # wB97X-3c is a composite -> empty basis string (its own def2-mTZVP+gCP+D4)
            e_wb = T.calc_orca_sp(xyz, "wB97X-3c", "", "DMSO",
                                  maxcore_mb=2000, orca_bin=ORCA, timeout=7200)
            rec["E_r2scan_Eh"] = e_r2
            rec["E_wb97x_Eh"] = e_wb
            if e_r2 is not None and e_wb is not None:
                rec["dE_wb97x_minus_r2scan_kcal"] = (e_wb - e_r2) * HK
            shutil.rmtree(wd, ignore_errors=True)
        except Exception as e:
            rec["error"] = str(e)[:100]
        print(f"  id={r.id} dE={rec.get('dE_wb97x_minus_r2scan_kcal')}", flush=True)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    shutil.rmtree(wroot, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
