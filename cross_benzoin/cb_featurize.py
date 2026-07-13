#!/usr/bin/env python3
"""
cross_benzoin unified featurizer  (see ARCHITECTURE.md)
=======================================================
One entry point for benzoin PRODUCT featurization (homo = diagonal special case).
Aldehyde and product share the SAME funnel_v3 method + descriptor backends, and
ALL geometries/energies/descriptors are saved and cross-linked by stable IDs.

Reuses (does not re-implement) the validated backends in pipeline/compute:
  ald_descriptors_qm (xTB/morfeus/Multiwfn), thermo_orca (ohess G),
  conf_funnel_v3 (conformer search), featurize_product (benzoin-core logic).

Inputs (one of):
  --pairs PAIRS.csv      columns: donor_id,acceptor_id,donor_smiles,acceptor_smiles
  --homo-from LIB.csv    aldehyde library (index,SMILES[,xtb_optimized]) -> homo pairs

Output: a RUN DIRECTORY with fixed names:
  <out>/aldehydes.csv  <out>/products.csv  <out>/xyz_ald/  <out>/xyz_prod/

Example:
  python cb_featurize.py --homo-from data/library/aldehydes_clean_v6.csv \
      --out data/cross_benzoin/homo_v6 --emit-aldehydes --multiwfn \
      --multiwfn-bin /home/schen3/mutiwfn/Multiwfn_noGUI --workers 12 --n-confs 10
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# make the validated backends importable
_COMPUTE = Path(__file__).resolve().parents[1] / "pipeline" / "compute"
sys.path.insert(0, str(_COMPUTE))
# conf_funnel_v3 MUST be imported BEFORE thermo_orca / ald_descriptors_qm: thermo_orca and
# conf_funnel_v2 are mutually circular, and loading conf_funnel_v3 first fully initialises
# thermo_orca via that path. The old order (ald/thermo first) crashed with
# "cannot import name '_mol_rotbonds' from partially initialized module 'thermo_orca'".
import conf_funnel_v3                   # noqa: E402
import ald_descriptors_qm as A          # noqa: E402
import thermo_orca as Th                # noqa: E402
import featurize_product as FP          # noqa: E402  (benzoin-core + product calc_* reuse)

from rdkit import Chem                  # noqa: E402

log = logging.getLogger("cb_featurize")
HARTREE_TO_KCAL = 627.509474

# ── g-xTB baseline, FUSED into this pass ─────────────────────────────────────
# The production Δ-model baseline is g-xTB. Computing it as a separate job would redo the
# whole funnel_v3 + GFN2-ohess geometry (the expensive part) a SECOND time. Instead we do
# one g-xTB COSMO(DMSO) SP on the GFN2-ohess geometry we already have → G_gxtb in the same
# pass (≈+10-20% vs ~2× cost). Mirrors pipeline/compute/gxtb_baseline.py. Needs GXTB_BIN
# (+ GXTB_SOLV) in the env, exactly as submit_gxtb_baseline.sh sets them.
XTB_GXTB = os.environ.get(
    "GXTB_BIN", "/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb")
GXTB_SOLV = os.environ.get("GXTB_SOLV", "cosmo dmso").split()  # ALPB/GBSA fatal; COSMO ~ CPCM
_GXTB_E = re.compile(r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh")


def _gxtb_sp(geom: Path, wd: Path, charge: int = 0, timeout: int = 900) -> float | None:
    """g-xTB COSMO(DMSO) single point on `geom` → total energy E_gxtb (Eh), None on failure."""
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "g.xyz").write_text(Path(geom).read_text())
    cmd = [XTB_GXTB, "g.xyz", "--gxtb", "--sp", "--chrg", str(charge)]
    if GXTB_SOLV:
        cmd += ["--" + GXTB_SOLV[0], *GXTB_SOLV[1:]]
    try:
        r = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    (wd / "gxtb_sp.log").write_text(r.stdout + r.stderr)
    m = _GXTB_E.findall(r.stdout + r.stderr)
    return float(m[-1]) if m else None


def _g_gxtb(ohess_stdout: str, ohess_dir: Path, smiles: str, wd: Path) -> float | None:
    """G_gxtb (Eh) = E_gxtb + (G_gfn2 − E_el_gfn2): g-xTB electronic energy on the GFN2-ohess
    geometry, reusing the GFN2 RRHO thermal correction. None if any piece is missing."""
    G = Th.parse_xtb_G(ohess_stdout)
    E_el = Th._parse_xtb_energy(ohess_stdout)
    geom = ohess_dir / "xtbopt.xyz"
    if G is None or E_el is None or not geom.exists():
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        chg = Chem.GetFormalCharge(m) if m is not None else 0
    except Exception:
        chg = 0
    E_gxtb = _gxtb_sp(geom, wd / "gxtb", charge=chg)
    return E_gxtb + (G - E_el) if E_gxtb is not None else None

# ── Schema (single source of truth) ─────────────────────────────────────────
_ALD_DESC = [c for c in A._ALL_FIELDS
             if c not in ("index", "SMILES", "PubChem_CID",
                          "xtb_optimized", "error", "xyz_file")]
ALD_FIELDS = ["id", "smiles", "xtb_optimized", "error", "xyz_file", "G_xtb", "G_gxtb"] + _ALD_DESC

_PROD_DESC = FP._XTB + FP._MORF + FP._MWF
PROD_FIELDS = (["id", "donor_id", "acceptor_id", "donor_smiles", "acceptor_smiles",
                "smiles", "reaction_type", "is_homo", "xtb_optimized", "error", "xyz_file"]
               + _PROD_DESC + ["G_donor", "G_acceptor", "G_xtb", "dG_xtb_kcal",
                               "G_donor_gxtb", "G_acceptor_gxtb", "G_gxtb", "dG_gxtb_kcal"])


def pair_id(did: str, aid: str) -> str:
    # homo (donor==acceptor) → single id, not the redundant "<id>__<id>"
    return did if did == aid else f"{did}__{aid}"


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))[:60]


def _rank(name: str):
    return conf_funnel_v3.rank_conformers_funnel_v3 if name == "funnel_v3" else Th._rank_conformers


# ── Aldehyde: funnel_v3 geometry (saved) + descriptors + G ───────────────────
def featurize_aldehyde(ald_id, smi, *, xyz_dir, work_dir, xtb_bin, mwf_bin, do_multiwfn,
                       solvent, n_confs, T, P, cores, jobs, timeout, conformer):
    row = {f: None for f in ALD_FIELDS}
    row.update({"id": str(ald_id), "smiles": smi, "xtb_optimized": False, "error": ""})
    wd = work_dir / f"ald_{_safe(ald_id)}"
    # Free this aldehyde's scratch (conf/desc/mwf/ohess — hundreds of tiny xTB+Multiwfn
    # files) before returning; row + saved xyz are all that's kept. work_dir is on the
    # shared gpfs:scratch1/nodespecific tree with ONE per-user inode quota across all
    # nodes, so without per-molecule cleanup a full 220k chunk array (esp. with Multiwfn,
    # and at high %throttle) exhausts it mid-run with Errno 122. The chunk-end rmtree is
    # too late — all molecules' scratch coexists until then.
    try:
        ranked = _rank(conformer)(smi, wd / "conf", xtb_bin, n_confs, "ald",
                                  solvent=solvent, cores=cores, workers=jobs)
        if not ranked:
            row["error"] = "ald_embed_failed"
            return row, None
        best = ranked[0][0]
        row["xtb_optimized"] = True
        xp = xyz_dir / f"ald_{_safe(ald_id)}.xyz"
        xp.write_text(best, encoding="utf-8")
        row["xyz_file"] = str(xp)
        sym, coord = A.parse_xyz(best)
        desc = {}
        desc.update(A.calc_xtb(best, sym, coord, xtb_bin, wd / "desc"))
        desc.update(A.calc_morfeus(sym, coord))
        if do_multiwfn and mwf_bin:
            desc.update(A.calc_multiwfn(best, sym, coord, xtb_bin, mwf_bin,
                                        wd / "mwf", stem=f"ald_{_safe(ald_id)}"))
        for k, v in desc.items():
            if k in row:
                row[k] = v
        sa, _ = Th.run_ohess(best, wd / "ohess", xtb_bin, T, P, solvent=solvent,
                             cores=cores, timeout=timeout)
        G = Th.parse_xtb_G(sa)
        row["G_xtb"] = G
        G_gxtb = _g_gxtb(sa, wd / "ohess", smi, wd)   # g-xTB SP on the same ohess geom
        row["G_gxtb"] = G_gxtb
        return row, (G, G_gxtb)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ── Product: build, funnel_v3 geometry (saved) + descriptors + ΔG ────────────
def featurize_pair(rec, *, g_cache, xyz_dir, work_dir, xtb_bin, mwf_bin, do_multiwfn,
                   solvent, n_confs, T, P, cores, jobs, timeout, conformer):
    did = str(rec.get("donor_id") or rec.get("index") or "d")
    aid = str(rec.get("acceptor_id") or did)
    donor = (rec.get("donor_smiles") or "").strip()
    acc = (rec.get("acceptor_smiles") or "").strip()
    pid = pair_id(did, aid)
    row = {f: None for f in PROD_FIELDS}
    row.update({"id": pid, "donor_id": did, "acceptor_id": aid,
                "donor_smiles": donor, "acceptor_smiles": acc,
                "xtb_optimized": False, "error": ""})
    if not donor or not acc:
        row["error"] = "missing_smiles"
        return row
    is_homo = Chem.CanonSmiles(donor) == Chem.CanonSmiles(acc)
    row["is_homo"] = is_homo
    row["reaction_type"] = FP.reaction_type(FP.classify(donor), FP.classify(acc), is_homo)
    prod = FP.build_product(donor, acc)
    if not prod:
        row["error"] = "product_build_failed"
        return row
    row["smiles"] = prod
    wd = work_dir / f"prod_{_safe(pid)}"
    # Per-product scratch cleanup — see featurize_aldehyde: bounds the live inode
    # footprint to ~workers concurrent molecules instead of a whole chunk, which is
    # what keeps a 220k funnel_v3 array under the shared nodespecific inode quota.
    try:
        ranked = _rank(conformer)(prod, wd / "conf", xtb_bin, n_confs, "prod",
                                  solvent=solvent, cores=cores, workers=jobs)
        if not ranked:
            row["error"] = "prod_embed_failed"
            return row
        best = ranked[0][0]
        row["xtb_optimized"] = True
        _stem = _safe(did) if did == aid else f"{_safe(did)}__{_safe(aid)}"
        xp = xyz_dir / f"prod_{_stem}.xyz"
        xp.write_text(best, encoding="utf-8")
        row["xyz_file"] = str(xp)
        sym, coord = A.parse_xyz(best)
        core = FP.find_benzoin_core(sym, coord)
        if core is None:
            row["error"] = "core_not_found"
            return row
        descs = [FP.calc_xtb_product(best, sym, coord, core, xtb_bin, wd / "desc"),
                 FP.calc_morfeus_product(sym, coord, core)]
        if do_multiwfn and mwf_bin:
            descs.append(FP.calc_multiwfn_product(best, sym, coord, core, xtb_bin, mwf_bin,
                                                  wd / "mwf", pid))
        for d in descs:
            for k, v in d.items():
                if k in row:
                    row[k] = v
        sp, _ = Th.run_ohess(best, wd / "ohess", xtb_bin, T, P, solvent=solvent,
                             cores=cores, timeout=timeout)
        Gp = Th.parse_xtb_G(sp)
        Gp_g = _g_gxtb(sp, wd / "ohess", prod, wd)            # product g-xTB G
        Gd, Gd_g = g_cache.get(Chem.CanonSmiles(donor)) or (None, None)
        Ga, Ga_g = g_cache.get(Chem.CanonSmiles(acc)) or (None, None)
        row["G_donor"], row["G_acceptor"], row["G_xtb"] = Gd, Ga, Gp
        row["G_donor_gxtb"], row["G_acceptor_gxtb"], row["G_gxtb"] = Gd_g, Ga_g, Gp_g
        if None not in (Gp, Gd, Ga):
            row["dG_xtb_kcal"] = round((Gp - Gd - Ga) * HARTREE_TO_KCAL, 4)
        else:
            row["error"] = (row["error"] + ";" if row["error"] else "") + "dG_failed"
        if None not in (Gp_g, Gd_g, Ga_g):
            row["dG_gxtb_kcal"] = round((Gp_g - Gd_g - Ga_g) * HARTREE_TO_KCAL, 4)
        return row
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ── Inputs ───────────────────────────────────────────────────────────────────
def load_pairs(args) -> list[dict]:
    if args.pairs:
        with open(args.pairs, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    rows = []
    with open(args.homo_from, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            smi = (r.get("SMILES") or r.get("smiles") or "").strip()
            if not smi:
                continue
            if "xtb_optimized" in r and str(r["xtb_optimized"]).strip() not in ("", "True", "true", "1"):
                continue
            i = str(r.get("index") or r.get("id") or len(rows))
            rows.append({"donor_id": i, "acceptor_id": i,
                         "donor_smiles": smi, "acceptor_smiles": smi})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pairs", help="pairs CSV: donor_id,acceptor_id,donor_smiles,acceptor_smiles")
    src.add_argument("--homo-from", help="aldehyde library CSV (index,SMILES) -> homo pairs")
    ap.add_argument("--out", required=True, help="run output directory")
    ap.add_argument("--emit-aldehydes", action="store_true",
                    help="also featurize+save each unique aldehyde (funnel_v3)")
    ap.add_argument("--xtb-bin", default=shutil.which("xtb") or "/home/schen3/xtb/bin/xtb")
    ap.add_argument("--multiwfn", action="store_true")
    ap.add_argument("--multiwfn-bin", default="/home/schen3/mutiwfn/Multiwfn_noGUI")
    ap.add_argument("--conformer", choices=["funnel_v3", "rank"], default="funnel_v3")
    ap.add_argument("--solvent", default="dmso")
    ap.add_argument("--n-confs", type=int, default=10)
    ap.add_argument("--xtb-cores", type=int, default=2)
    ap.add_argument("--parallel-jobs", type=int, default=1)
    ap.add_argument("--ohess-timeout", type=int, default=900)
    ap.add_argument("--T", type=float, default=298.15)
    ap.add_argument("--P", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    xtb_bin = shutil.which(args.xtb_bin) or args.xtb_bin
    solvent = "" if args.solvent.lower() == "none" else args.solvent
    do_multiwfn = args.multiwfn and bool(args.multiwfn_bin)

    pairs = load_pairs(args)
    if args.max:
        pairs = pairs[: args.max]

    out = Path(args.out)
    xyz_ald, xyz_prod = out / "xyz_ald", out / "xyz_prod"
    work_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"cb_featurize_{os.getpid()}"
    for d in (out, xyz_prod, work_dir):
        d.mkdir(parents=True, exist_ok=True)
    if args.emit_aldehydes:
        xyz_ald.mkdir(parents=True, exist_ok=True)
    log.info("pairs=%d  emit_aldehydes=%s  multiwfn=%s  conformer=%s  out=%s",
             len(pairs), args.emit_aldehydes, do_multiwfn, args.conformer, out)

    # unique aldehydes: CanonSmiles -> (ald_id, smiles)
    uniq: dict[str, tuple[str, str]] = {}
    for r in pairs:
        for role in ("donor", "acceptor"):
            smi = (r.get(f"{role}_smiles") or "").strip()
            if not smi or not Chem.MolFromSmiles(smi):
                continue
            uniq.setdefault(Chem.CanonSmiles(smi), (str(r.get(f"{role}_id") or "a"), smi))

    g_cache: dict[str, float | None] = {}

    # ── aldehyde phase ────────────────────────────────────────────────────────
    akw = dict(xyz_dir=xyz_ald, work_dir=work_dir, xtb_bin=xtb_bin, mwf_bin=args.multiwfn_bin,
               do_multiwfn=do_multiwfn, solvent=solvent, n_confs=args.n_confs, T=args.T, P=args.P,
               cores=args.xtb_cores, jobs=args.parallel_jobs, timeout=args.ohess_timeout,
               conformer=args.conformer)
    items = list(uniq.items())  # [(canon, (id, smi)), ...]
    log.info("unique aldehydes: %d", len(items))
    if args.emit_aldehydes:
        with open(out / "aldehydes.csv", "w", newline="", encoding="utf-8") as afh:
            aw = csv.DictWriter(afh, fieldnames=ALD_FIELDS, extrasaction="ignore")
            aw.writeheader()
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(featurize_aldehyde, aid, smi, **akw): canon
                        for canon, (aid, smi) in items}
                for n, fut in enumerate(as_completed(futs), 1):
                    canon = futs[fut]
                    try:
                        prow, gpair = fut.result()      # gpair = (G_gfn2, G_gxtb)
                    except Exception as exc:
                        prow, gpair = {"smiles": canon, "error": f"exception:{exc}"}, None
                    aw.writerow(prow); afh.flush(); g_cache[canon] = gpair
                    if n % 25 == 0 or n == len(items):
                        log.info("  ald %d/%d", n, len(items))
    else:  # only need G for ΔG (GFN2 only; g-xTB unavailable on this path → None)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(FP.ald_free_energy, smi, work_dir / f"aldG_{_safe(aid)}", xtb_bin,
                              solvent, args.n_confs, args.T, args.P, args.xtb_cores,
                              args.parallel_jobs, args.ohess_timeout, args.conformer): canon
                    for canon, (aid, smi) in items}
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    res = fut.result()                  # ald_free_energy → (G, xyz)
                    g_cache[futs[fut]] = (res[0] if isinstance(res, tuple) else res, None)
                except Exception:
                    g_cache[futs[fut]] = None

    # ── product phase ─────────────────────────────────────────────────────────
    pkw = dict(g_cache=g_cache, xyz_dir=xyz_prod, work_dir=work_dir, xtb_bin=xtb_bin,
               mwf_bin=args.multiwfn_bin, do_multiwfn=do_multiwfn, solvent=solvent,
               n_confs=args.n_confs, T=args.T, P=args.P, cores=args.xtb_cores,
               jobs=args.parallel_jobs, timeout=args.ohess_timeout, conformer=args.conformer)
    n_ok = n_dg = n_err = 0
    with open(out / "products.csv", "w", newline="", encoding="utf-8") as pfh:
        pw = csv.DictWriter(pfh, fieldnames=PROD_FIELDS, extrasaction="ignore")
        pw.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(featurize_pair, r, **pkw): i for i, r in enumerate(pairs)}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {"id": str(futs[fut]), "error": f"exception:{exc}"}
                pw.writerow(row); pfh.flush()
                n_ok += 1; n_dg += row.get("dG_xtb_kcal") is not None; n_err += bool(row.get("error"))
    shutil.rmtree(work_dir, ignore_errors=True)
    log.info("done: %d products (%d with dG, %d errors) -> %s/products.csv", n_ok, n_dg, n_err, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
