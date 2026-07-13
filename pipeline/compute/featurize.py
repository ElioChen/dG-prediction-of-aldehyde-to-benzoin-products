#!/usr/bin/env python3
"""
Unified per-molecule featurization: descriptors + ΔG on ONE shared geometry.

Both ald_descriptors.py and thermo_orca.py used to generate & xTB-optimise the
aldehyde conformers independently — duplicate work, and (worse) the descriptors
could land on a different conformer than the ΔG label, adding train/label noise.

Here the aldehyde conformer search + xTB-opt runs ONCE; its best conformer feeds
both the descriptors and the aldehyde thermochemistry, and (with --ensemble-k>1)
the top-K conformers are Boltzmann-averaged for the DFT ΔG. One row out carries
the ~63 descriptors AND dG_xtb_kcal / dG_orca_kcal, geometry-consistent.

  python featurize.py --input mols.csv --output feats.csv \
      --xtb-bin xtb --multiwfn-bin Multiwfn_noGUI \
      --sp-method PBE0-D4 --sp-basis def2-TZVP --solvent dmso \
      --ensemble-k 5 --orca-nprocs 16 --workers 1

Designed to be driven by a SLURM array (one molecule per task) — see
pipeline/slurm/submit_featurize_array.sh.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import ald_descriptors as A
import thermo_orca as Th

log = logging.getLogger("featurize")
OUT_FIELDS = list(A._ALL_FIELDS) + ["dG_xtb_kcal", "dG_orca_kcal"]


def _thermo_ensemble(confs, work_dir: Path, *, T, kw, jobs: int = 1):
    """Run ohess+ORCA on each (xyz, E) conformer; return (G_best, G_orca_ensemble).

    Conformers are independent, so up to `jobs` run concurrently (each using
    xtb_cores for the Hessian + orca_nprocs for the single-point). G_best = the
    rank-0 (lowest-xTB) conformer's free energy → dG_xtb (single, matches
    inference). G_orca = Boltzmann ensemble of E_orca + (G_xtb − E_el_xtb).
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(i_xyz):
        i, (xyz, _) = i_xyz
        return i, Th._ohess_orca(xyz, work_dir / f"c{i:02d}", **kw)

    items = list(enumerate(confs))
    if jobs > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            out = list(ex.map(_one, items))
    else:
        out = [_one(t) for t in items]

    G_best = None
    comps: list[float] = []
    for i, (G, E_el, E_orca, _) in sorted(out):     # i=0 is the lowest-xTB conformer
        if G is None:
            continue
        if G_best is None:
            G_best = G
        if E_orca is not None and E_el is not None:
            comps.append(E_orca + (G - E_el))
    return G_best, (Th._ensemble_free_energy(comps, T) if comps else None)


def _select(ranked, ensemble_k: int, window_kcal: float):
    """Top-K conformers within an xTB energy window (single best when k<=1)."""
    if not ranked:
        return []
    if ensemble_k <= 1:
        return ranked[:1]
    E0 = ranked[0][1]
    win = window_kcal / Th.HARTREE_TO_KCAL
    return [(x, e) for x, e in ranked[:ensemble_k] if e - E0 <= win]


def featurize_one(rec: dict, idx: int, *, work_dir: Path, xyz_dir: Path,
                  xtb_bin: str, mwf_bin: str | None, do_multiwfn: bool,
                  solvent: str, n_confs: int, ensemble_k: int, ensemble_window: float,
                  sp_method: str | None, sp_basis: str, orca_bin: str,
                  orca_nprocs: int, orca_maxcore: int, T: float, P_atm: float,
                  max_atoms: int, xtb_cores: int = 2,
                  parallel_jobs: int = 4) -> dict[str, Any]:
    index_val = str(rec.get("index") or idx).strip()
    ald_smi = (rec.get("SMILES") or rec.get("aldehyde_smiles") or "").strip()
    row: dict[str, Any] = {f: None for f in OUT_FIELDS}
    row.update({"index": index_val, "SMILES": ald_smi,
                "PubChem_CID": str(rec.get("PubChem_CID") or "").strip(),
                "xtb_optimized": False, "error": "", "xyz_file": ""})
    stem = f"m{idx:06d}"
    mdir = work_dir / stem
    orca_solvent = Th._ORCA_SOLVENT.get(solvent.lower(), "") if solvent else ""
    okw = dict(xtb_bin=xtb_bin, T=T, P_atm=P_atm, solvent=solvent,
               sp_method=sp_method, sp_basis=sp_basis, orca_bin=orca_bin,
               orca_nprocs=orca_nprocs, orca_maxcore=orca_maxcore,
               orca_solvent=orca_solvent, max_atoms=max_atoms, xtb_cores=xtb_cores)

    # ── 0. RDKit 2D descriptors (geometry-free) ───────────────────────────
    row.update(A.calc_rdkit(ald_smi))

    # ── 1. SHARED: aldehyde conformer search + xTB-opt (ONCE, parallel) ───
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

    # ── 2a. Descriptors on the shared best aldehyde geometry ──────────────
    symbols, coords = A.parse_xyz(best_xyz)
    row.update(A.calc_xtb(best_xyz, symbols, coords, xtb_bin, mdir / "desc_xtb"))
    row.update(A.calc_morfeus(symbols, coords))
    if do_multiwfn and mwf_bin:
        row.update(A.calc_multiwfn(best_xyz, symbols, coords, xtb_bin, mwf_bin,
                                   mdir / "mwf", stem=stem))

    # ── 2b. Thermochemistry on the SAME ranked conformers ─────────────────
    # dG_xtb (the model feature) is xTB-only and is computed whether or not a DFT
    # method is set — so inference (sp_method=None) reuses this exact path and gets
    # descriptors + dG_xtb on one shared geometry, identical to training.
    bz_smi = Th._make_benzoin_smiles(ald_smi)
    if not bz_smi:
        row["error"] = "benzoin_smarts_failed"
        return row
    sel_ald = _select(ranked, ensemble_k, ensemble_window)
    G_ald, Gorca_ald = _thermo_ensemble(sel_ald, mdir / "ald_thermo",
                                         T=T, kw=okw, jobs=parallel_jobs)
    ranked_bz = Th._rank_conformers(bz_smi, mdir / "bz_conf", xtb_bin,
                                    n_confs, "bz", solvent=solvent,
                                    cores=xtb_cores, workers=parallel_jobs)
    sel_bz = _select(ranked_bz, ensemble_k, ensemble_window)
    G_bz, Gorca_bz = _thermo_ensemble(sel_bz, mdir / "bz_thermo",
                                      T=T, kw=okw, jobs=parallel_jobs)

    row["dG_xtb_kcal"] = Th._dG_kcal(G_bz, G_ald)            # xТB ΔG (single best)
    if sp_method:
        row["dG_orca_kcal"] = Th._dG_kcal(Gorca_bz, Gorca_ald)  # ensemble DFT-level
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--xtb-bin", default=shutil.which("xtb"))
    ap.add_argument("--multiwfn-bin", default=None)
    ap.add_argument("--no-multiwfn", action="store_true")
    ap.add_argument("--solvent", default="dmso")
    ap.add_argument("--n-confs", type=int, default=10)
    ap.add_argument("--ensemble-k", type=int, default=1)
    ap.add_argument("--ensemble-window", type=float, default=3.0)
    ap.add_argument("--sp-method", default="PBE0-D4")
    ap.add_argument("--sp-basis", default="def2-TZVP")
    ap.add_argument("--orca-bin", default=Th._DEFAULT_ORCA)
    ap.add_argument("--orca-nprocs", type=int, default=4,
                    help="cores per ORCA single-point (4-16)")
    ap.add_argument("--orca-maxcore", type=int, default=1800)
    ap.add_argument("--xtb-cores", type=int, default=2,
                    help="threads per xTB call (2-4; xTB scales poorly beyond)")
    ap.add_argument("--parallel-jobs", type=int, default=4,
                    help="conformer sub-jobs run concurrently within the task "
                         "(parallel_jobs x orca_nprocs should fit --cpus-per-task)")
    ap.add_argument("--max-atoms", type=int, default=150)
    ap.add_argument("--T", type=float, default=298.15)
    ap.add_argument("--P", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    work_dir = Path(args.work_dir or os.environ.get("TMPDIR", "/tmp")) / "featurize"
    xyz_dir = Path(args.output).parent / "xyz"
    work_dir.mkdir(parents=True, exist_ok=True)
    xyz_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input) as fh:
        records = list(csv.DictReader(fh))
    log.info("Loaded %d molecules from %s", len(records), args.input)

    kw = dict(work_dir=work_dir, xyz_dir=xyz_dir, xtb_bin=args.xtb_bin,
              mwf_bin=args.multiwfn_bin, do_multiwfn=not args.no_multiwfn,
              solvent=args.solvent, n_confs=args.n_confs,
              ensemble_k=args.ensemble_k, ensemble_window=args.ensemble_window,
              sp_method=args.sp_method, sp_basis=args.sp_basis,
              orca_bin=args.orca_bin, orca_nprocs=args.orca_nprocs,
              orca_maxcore=args.orca_maxcore, T=args.T, P_atm=args.P,
              max_atoms=args.max_atoms, xtb_cores=args.xtb_cores,
              parallel_jobs=args.parallel_jobs)

    results: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(featurize_one, r, i, **kw): i
                    for i, r in enumerate(records)}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    log.error("worker %d failed: %s", futs[fut], exc)
    else:
        for i, r in enumerate(records):
            try:
                row = featurize_one(r, i, **kw)
            except Exception as exc:
                row = {f: None for f in OUT_FIELDS}
                row.update({"index": r.get("index"), "SMILES": r.get("SMILES"),
                            "error": f"exception:{exc}"})
            results.append(row)
            dg = row.get("dG_orca_kcal")
            log.info("[%s] dG_orca=%s err=%s",
                     row.get("index"), f"{dg:+.2f}" if dg is not None else "—",
                     row.get("error") or "")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    n_ok = sum(1 for r in results if r.get("dG_orca_kcal") is not None)
    log.info("Wrote %s — %d rows, %d with DFT ΔG", args.output, len(results), n_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
