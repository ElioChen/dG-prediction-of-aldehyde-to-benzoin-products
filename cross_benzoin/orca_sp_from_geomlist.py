#!/usr/bin/env python
"""r2SCAN-3c/CPCM(DMSO) single points over a flat (id, xyz_path) work list.

Purpose-built for the rounds 8-9 DFT-label recovery (see
docs/HANDOFF_round10_20260721_ZH.md and RECOVERY_REPORT_20260902.md). The
2026-07 Snellius purge took `data/raw/dft_sp_cross/cross_round{8,9}_dft_sp.csv`
AND the homo full-library aldehyde SP cache `data/raw/dft_sp_funnelv3/`, so
`dft_sp_cross_from_geom.py` (which assumes the aldehyde side is already cached)
cannot run for r8/9. Instead we recompute all three species fresh:

    dG_orca = G(product) - G(donor) - G(acceptor)
    G(species) = E_orca_SP(species) + (G_xtb(species) - E_el_xtb(species))

This script does ONLY the E_orca_SP leg for whichever species list it is given
(products or aldehydes); the xTB RRHO thermal term is carried separately and the
per-pair dG is assembled by assemble_r89_dft_sp.py.

Input CSV columns: `id`, `xyz_path` (absolute). Output CSV columns:
`id`, `E_orca_Eh`, `error`. Resume-safe: rows already present in --out-csv with a
non-empty E_orca_Eh are skipped; each result is flushed as it completes. Use
--skip/--max to run as a CHUNK-based SLURM array (slurm/submit_r89_sp_array.sh).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "compute"))
import conf_funnel_v3  # noqa: F401,E402  # must precede thermo_orca (breaks a circular import)
import thermo_orca as T  # noqa: E402


def _sp_one(args) -> dict:
    rid, xyz, method, basis, osolv, maxcore, orca_bin, timeout = args
    p = Path(xyz)
    if not p.exists():
        return {"id": rid, "E_orca_Eh": "", "error": "xyz_missing"}
    wd = Path(tempfile.mkdtemp(prefix="r89sp_", dir=os.environ.get("TMPDIR", "/tmp")))
    try:
        local = wd / "mol.xyz"
        shutil.copy(p, local)
        E = T.calc_orca_sp(local, method, basis, osolv, maxcore_mb=maxcore,
                           orca_bin=orca_bin, timeout=timeout)
        if E is None:
            return {"id": rid, "E_orca_Eh": "", "error": "orca_sp_failed"}
        return {"id": rid, "E_orca_Eh": f"{E:.10f}", "error": ""}
    except Exception as exc:  # noqa: BLE001 -- one bad row must not kill the chunk
        return {"id": rid, "E_orca_Eh": "", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geom-list", required=True, help="CSV with columns id,xyz_path")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=10**9)
    ap.add_argument("--method", default="r2SCAN-3c")
    ap.add_argument("--basis", default="def2-mTZVP")   # ignored for the -3c composite
    ap.add_argument("--solvent", default="dmso")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--maxcore", type=int, default=1500)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--orca-bin", default="/home/schen3/orca/orca")
    ap.add_argument("--smoke", action="store_true", help="only the first 3 rows of the slice")
    args = ap.parse_args()

    with open(args.geom_list, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("id") and r.get("xyz_path")]
    sl = rows[args.skip: args.skip + args.max]
    if args.smoke:
        sl = sl[:3]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8-sig", newline="") as fh:
            done = {r["id"] for r in csv.DictReader(fh) if r.get("E_orca_Eh")}
    todo = [r for r in sl if r["id"] not in done]
    print(f"orca_sp_from_geomlist: {len(todo)} SPs "
          f"(slice {args.skip}:{args.skip + args.max}, {len(done)} already done, "
          f"workers={args.workers}, {args.method}/CPCM({args.solvent}))", flush=True)

    osolv = T._ORCA_SOLVENT.get(args.solvent.lower(), "DMSO")
    tasks = [(r["id"], r["xyz_path"], args.method, args.basis, osolv,
              args.maxcore, args.orca_bin, args.timeout) for r in todo]

    write_header = not (out_path.exists() and out_path.stat().st_size > 0)
    n_ok = 0
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "E_orca_Eh", "error"])
        if write_header:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed(ex.submit(_sp_one, t) for t in tasks):
                rec = fut.result()
                w.writerow(rec)
                fh.flush()
                n_ok += bool(rec["E_orca_Eh"])
    print(f"done: {n_ok}/{len(todo)} ok -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
