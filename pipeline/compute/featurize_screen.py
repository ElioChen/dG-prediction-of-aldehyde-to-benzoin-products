#!/usr/bin/env python3
"""
Library screening featurizer — QM descriptors + xTB ΔG only (NO DFT/ORCA).
==========================================================================
A stripped-down sibling of featurize.py for screening the WHOLE filter_v5 library
(data/library/aldehydes_clean_v5.csv, ~221k aldehydes) cheaply, then looking at the
descriptor / ΔG distributions. Differences vs featurize.py:

  • descriptors come from ald_descriptors_qm (Level 0 RDKit-2D block removed) — only
    the geometry/QM levels: xTB (L1), morfeus (L2), optional Multiwfn (L3);
  • ΔG is xTB-only: GFN2-xTB --ohess (ALPB solvent) free energies on the rank-0
    conformer of aldehyde and benzoin product. No ORCA single-point, no Shermo.
    → one row per aldehyde: the QM descriptors + dG_xtb_kcal.

The shared aldehyde conformer (best xTB-opt geometry) feeds BOTH the descriptors and
the aldehyde free energy, so they stay geometry-consistent (same rationale as
featurize.py). Reaction:  2 R-CHO → R-CH(OH)-C(=O)-R.

Multiwfn (L3 ADCH/QTAIM) is OFF by default — it is ~minutes/molecule and infeasible
across 221k; enable with --multiwfn for a small subset. Without it the adch_*/qtaim_*
columns are written empty (schema stays fixed = ald_descriptors_qm._ALL_FIELDS).

Usage
-----
  # quick local test (10 built-in-style molecules via --smiles)
  python featurize_screen.py --smiles "O=Cc1ccccc1" --smiles "O=Cc1ccco1" \
         --output /tmp/screen.csv --xtb-bin /home/schen3/xtb/bin/xtb

  # a chunk of the v5 library (driven by a SLURM array — see
  # pipeline/slurm/submit_featurize_screen_array.sh)
  python featurize_screen.py --input chunk.csv --output out.csv \
         --xtb-bin xtb --solvent dmso --n-confs 10 --xtb-cores 4
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import ald_descriptors_qm as A
import thermo_orca as Th

log = logging.getLogger("featurize_screen")

# Fixed output schema: every QM descriptor column + the single xTB ΔG.
OUT_FIELDS = list(A._ALL_FIELDS) + ["dG_xtb_kcal"]


def featurize_one(
    rec: dict, idx: int, *, work_dir: Path, xyz_dir: Path,
    xtb_bin: str, mwf_bin: str | None, do_multiwfn: bool,
    solvent: str, n_confs: int, T: float, P_atm: float,
    xtb_cores: int = 2, parallel_jobs: int = 4, ohess_timeout: int = 900,
) -> dict[str, Any]:
    index_val = str(rec.get("index") or idx).strip()
    ald_smi = (rec.get("SMILES") or rec.get("aldehyde_smiles") or "").strip()
    row: dict[str, Any] = {f: None for f in OUT_FIELDS}
    row.update({"index": index_val, "SMILES": ald_smi,
                "PubChem_CID": str(rec.get("PubChem_CID") or "").strip(),
                "xtb_optimized": False, "error": "", "xyz_file": ""})
    if not ald_smi:
        row["error"] = "no_smiles"
        return row
    stem = f"m{idx:06d}"
    mdir = work_dir / stem
    # Free this molecule's conformer/xTB scratch (ald_conf, bz_conf, *_ohess, desc_xtb —
    # each hundreds of tiny xTB files) before returning: only `row` + the saved xyz are
    # kept. Without this every molecule's scratch lives for the whole chunk task, and since
    # work_dir is on the SHARED gpfs:scratch1/nodespecific tree (one per-user inode quota
    # across all nodes), a 220k screen at --array=...%128 would exhaust it mid-run with
    # Errno 122 (exactly what sank prod_bvalid at %8). rmtree in finally = footprint bounded
    # to ~workers concurrent molecules instead of a whole chunk.
    try:
        # ── 1. SHARED: aldehyde conformer search + xTB-opt (best geometry) ────────
        ranked = Th._rank_conformers(ald_smi, mdir / "ald_conf", xtb_bin,
                                     n_confs, "ald", solvent=solvent,
                                     cores=xtb_cores, workers=parallel_jobs)
        if not ranked:
            row["error"] = "ald_embed_failed"
            return row
        best_xyz = ranked[0][0]
        row["xtb_optimized"] = True
        xyz_out = xyz_dir / f"{stem}.xyz"
        xyz_out.write_text(best_xyz, encoding="utf-8")
        row["xyz_file"] = str(xyz_out)

        # ── 2. QM descriptors on the shared best aldehyde geometry ───────────────
        symbols, coords = A.parse_xyz(best_xyz)
        row.update(A.calc_xtb(best_xyz, symbols, coords, xtb_bin, mdir / "desc_xtb"))
        row.update(A.calc_morfeus(symbols, coords))
        if do_multiwfn and mwf_bin:
            row.update(A.calc_multiwfn(best_xyz, symbols, coords, xtb_bin, mwf_bin,
                                       mdir / "mwf", stem=stem))

        # ── 3. xTB ohess free energy of the aldehyde (rank-0 conformer) ──────────
        sa, _ = Th.run_ohess(best_xyz, mdir / "ald_ohess", xtb_bin, T, P_atm,
                             solvent=solvent, cores=xtb_cores, timeout=ohess_timeout)
        G_ald = Th.parse_xtb_G(sa)

        # ── 4. benzoin product: conformer search + ohess ─────────────────────────
        bz_smi = Th._make_benzoin_smiles(ald_smi)
        if not bz_smi:
            row["error"] = "benzoin_smarts_failed"
            return row
        ranked_bz = Th._rank_conformers(bz_smi, mdir / "bz_conf", xtb_bin,
                                        n_confs, "bz", solvent=solvent,
                                        cores=xtb_cores, workers=parallel_jobs)
        if not ranked_bz:
            row["error"] = "bz_embed_failed"
            return row
        sb, _ = Th.run_ohess(ranked_bz[0][0], mdir / "bz_ohess", xtb_bin, T, P_atm,
                             solvent=solvent, cores=xtb_cores, timeout=ohess_timeout)
        G_bz = Th.parse_xtb_G(sb)

        # ── 5. ΔG (xTB ohess + ALPB), single best conformer each side ────────────
        row["dG_xtb_kcal"] = Th._dG_kcal(G_bz, G_ald)
        if row["dG_xtb_kcal"] is None:
            row["error"] = (row["error"] + ";" if row["error"] else "") + "dG_xtb_failed"
        return row
    finally:
        shutil.rmtree(mdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None,
                    help="CSV with a SMILES column (default: aldehydes_clean_v5.csv path)")
    ap.add_argument("--smiles", action="append", metavar="SMILES",
                    help="featurize these SMILES directly (repeatable) instead of --input")
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", default=None,
                    help="scratch dir (default: $TMPDIR or /tmp)/featurize_screen")
    ap.add_argument("--xtb-bin", default=shutil.which("xtb") or "/home/schen3/xtb/bin/xtb")
    ap.add_argument("--multiwfn-bin", default=None)
    ap.add_argument("--multiwfn", action="store_true",
                    help="enable Multiwfn L3 ADCH/QTAIM descriptors (slow; small subsets only)")
    ap.add_argument("--solvent", default="dmso",
                    help="xTB ALPB solvent (default dmso; 'none' = gas phase)")
    ap.add_argument("--n-confs", type=int, default=10,
                    help="max ETKDG conformers (auto-scaled by RotBonds; rigid → 1)")
    ap.add_argument("--xtb-cores", type=int, default=2,
                    help="threads per xTB call (2-4; xTB scales poorly beyond)")
    ap.add_argument("--parallel-jobs", type=int, default=4,
                    help="conformers optimised concurrently within a molecule")
    ap.add_argument("--ohess-timeout", type=int, default=900,
                    help="per-call xTB --ohess timeout (s); slow molecules fail fast "
                         "as dG_xtb_failed instead of blocking a worker (default 900)")
    ap.add_argument("--T", type=float, default=298.15)
    ap.add_argument("--P", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="molecules processed concurrently (process pool)")
    ap.add_argument("--skip", type=int, default=0, help="skip first N records")
    ap.add_argument("--max", type=int, default=0, help="cap at N records (0 = all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    xtb_bin = shutil.which(args.xtb_bin) or args.xtb_bin
    if not (Path(xtb_bin).exists() or shutil.which(xtb_bin)):
        log.error("xtb not found: %s", xtb_bin)
        return 1
    solvent = "" if args.solvent.lower() == "none" else args.solvent
    do_multiwfn = args.multiwfn and bool(args.multiwfn_bin)
    log.info("xtb=%s  solvent=%s  n_confs=%d  multiwfn=%s",
             xtb_bin, solvent or "gas", args.n_confs, do_multiwfn)

    # ── Records ───────────────────────────────────────────────────────────────
    if args.smiles:
        records = [{"index": str(i), "SMILES": s} for i, s in enumerate(args.smiles)]
    else:
        repo = Path(__file__).resolve().parents[2]
        inp = Path(args.input) if args.input else repo / "data/library/aldehydes_clean_v5.csv"
        if not inp.exists():
            log.error("Input not found: %s (use --smiles for a quick test)", inp)
            return 1
        with inp.open(encoding="utf-8") as fh:
            records = list(csv.DictReader(fh))
        log.info("Loaded %d records from %s", len(records), inp)
    if args.skip:
        records = records[args.skip:]
    if args.max:
        records = records[: args.max]
    if args.skip or args.max:
        log.info("Using %d records (skip=%d, max=%d)", len(records), args.skip, args.max)

    work_dir = Path(args.work_dir or os.environ.get("TMPDIR", "/tmp")) / "featurize_screen"
    xyz_dir = Path(args.output).parent / "xyz"
    work_dir.mkdir(parents=True, exist_ok=True)
    xyz_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    kw = dict(work_dir=work_dir, xyz_dir=xyz_dir, xtb_bin=xtb_bin,
              mwf_bin=args.multiwfn_bin, do_multiwfn=do_multiwfn,
              solvent=solvent, n_confs=args.n_confs, T=args.T, P_atm=args.P,
              xtb_cores=args.xtb_cores, parallel_jobs=args.parallel_jobs,
              ohess_timeout=args.ohess_timeout)

    # ── Stream rows to CSV (robust for long runs) ────────────────────────────
    n_ok = n_dg = n_err = 0
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()

        def _emit(row: dict) -> None:
            nonlocal n_ok, n_dg, n_err
            w.writerow(row); fh.flush()
            if row.get("error"):
                n_err += 1
            if row.get("dG_xtb_kcal") is not None:
                n_dg += 1
            n_ok += 1

        if args.workers > 1:
            import time
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(featurize_one, r, i, **kw): i
                        for i, r in enumerate(records)}
                done = 0
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:
                        log.error("worker %d failed: %s", i, exc)
                        row = {"index": str(i), "error": f"exception:{exc}"}
                    _emit(row)
                    done += 1
                    dg = row.get("dG_xtb_kcal")
                    log.info("[%d/%d %.1fs/mol] idx=%s dG_xtb=%s err=%s",
                             done, len(records), (time.time() - t0) / done,
                             row.get("index"), f"{dg:+.2f}" if dg is not None else "—",
                             row.get("error") or "")
        else:
            for i, r in enumerate(records):
                try:
                    row = featurize_one(r, i, **kw)
                except Exception as exc:
                    row = {"index": str(r.get("index") or i),
                           "SMILES": r.get("SMILES", ""), "error": f"exception:{exc}"}
                _emit(row)
                dg = row.get("dG_xtb_kcal")
                log.info("[%s] dG_xtb=%s err=%s", row.get("index"),
                         f"{dg:+.2f}" if dg is not None else "—", row.get("error") or "")

    log.info("Wrote %s — %d rows, %d with dG_xtb, %d with errors",
             args.output, n_ok, n_dg, n_err)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
