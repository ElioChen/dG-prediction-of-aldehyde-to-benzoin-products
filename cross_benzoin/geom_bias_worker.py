#!/usr/bin/env python
"""Geometry-source bias probe (product side).

For each heteroatom-rich cross-benzoin product: r2SCAN-3c SP on the existing
GFN2-xTB-opt geometry, then re-optimise the same structure with g-xTB (--gxtb --opt)
and r2SCAN-3c SP again. delta_E_prod = E(gxtb-geom) - E(gfn2-geom) [kcal/mol] is the
product-side geometry-bias contribution to that pair's dG label (donor/acceptor held
fixed -> d(dG) = d(E_prod)). A systematic non-zero delta for B / hypervalent-S / P
products would mean the GFN2 geometries bias the r2SCAN-3c labels for that subclass.

  /home/schen3/venv/nhc-workflow/bin/python cross_benzoin/geom_bias_worker.py \
      --sample data/cross_benzoin/geom_bias_sample.csv \
      --geom-dir data/cross_benzoin/geom_bias/geoms \
      --skip 0 --max 2 --out <chunk.csv> --scratch /scratch-local/...
"""
from __future__ import annotations
import argparse, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "/gpfs/scratch1/shared/schen3/benzoin-dg-restored/pipeline/compute")
import conf_funnel_v3  # noqa: F401  (break funnel<->thermo circular import)
import thermo_orca as T

HK = 627.5094740631
XTB = os.environ.get("XTB_BIN", "/home/schen3/xtb/bin/xtb")
ORCA = "/home/schen3/orca/orca"


def _gxtb_opt(xyz_str: str, wd: Path) -> str | None:
    """g-xTB geometry optimisation -> optimised xyz string (None on failure)."""
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "in.xyz").write_text(xyz_str)
    import subprocess
    r = subprocess.run([XTB, "in.xyz", "--gxtb", "--opt", "tight", "--chrg", "0"],
                       cwd=wd, capture_output=True, text=True, timeout=3600)
    (wd / "gxtb_opt.log").write_text(r.stdout + r.stderr)
    for cand in ("xtbopt.xyz", "in.xtbopt.xyz"):
        p = wd / cand
        if p.exists():
            return p.read_text()
    return None


def _sp(xyz_path: Path, wd: Path) -> float | None:
    return T.calc_orca_sp(xyz_path, "r2SCAN-3c", "def2-mTZVP", "DMSO",
                          maxcore_mb=2000, orca_bin=ORCA, timeout=5400)


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
    wroot = Path(a.scratch) / f"gb_{a.skip}"; wroot.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in df.itertuples():
        rec = {"id": r.id, "smiles": getattr(r, "smiles", None), "error": None}
        try:
            gfn2_xyz = gd / f"prod_{r.id}.xyz"
            if not gfn2_xyz.exists():
                rec["error"] = "no gfn2 geom"; rows.append(rec); continue
            wd = wroot / f"m_{abs(hash(r.id)) % 10**8}"; wd.mkdir(parents=True, exist_ok=True)

            e_gfn2 = _sp(gfn2_xyz, wd / "sp_gfn2")
            gxtb_xyz_str = _gxtb_opt(gfn2_xyz.read_text(), wd / "gxtb_opt")
            e_gxtb = None
            if gxtb_xyz_str:
                (wd / "gxtb.xyz").write_text(gxtb_xyz_str)
                e_gxtb = _sp(wd / "gxtb.xyz", wd / "sp_gxtb")

            rec["E_gfn2geom_Eh"] = e_gfn2
            rec["E_gxtbgeom_Eh"] = e_gxtb
            if e_gfn2 is not None and e_gxtb is not None:
                rec["delta_E_prod_kcal"] = (e_gxtb - e_gfn2) * HK
            shutil.rmtree(wd, ignore_errors=True)
        except Exception as e:
            rec["error"] = str(e)[:100]
        print(f"  id={r.id} dE_prod={rec.get('delta_E_prod_kcal')}", flush=True)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    shutil.rmtree(wroot, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
