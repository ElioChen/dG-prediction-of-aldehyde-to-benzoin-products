#!/usr/bin/env python
"""DRAFT — efficient full-library r2SCAN-3c ΔG by SP on SAVED funnel_v3 geometries.

The cross-benzoin run already produced, per molecule, the funnel_v3 reactant + product
geometries (homo_v6/chunk_*/xyz_ald, xyz_prod) AND the xTB free energies. So a DFT ΔG no
longer needs the conformer search + xTB-opt (the bulk of per-molecule time) — we only need
the r2SCAN-3c CPCM(DMSO) single point on each saved geometry and can REUSE the xTB RRHO
thermal correction:

    G_orca(species) = E_orca_SP(species)               # new, on the saved geom
                    + ( G_xtb(species) - E_el_xtb(species) )   # reuse, from the CSVs
    dG_orca = G_orca(product) - 2*G_orca(aldehyde)      # homo (donor==acceptor)

This is geometry-consistent with the g-xTB/GFN2 ΔG already in the library, and ~2x cheaper
than re-running the full thermo_orca pipeline (no ETKDG/xTB-opt).

⚠️ DRAFT — validate before a full launch: run --smoke on a few ids and confirm dG_orca
reproduces the existing dft_sp_r2scan3c_full values within method/geometry noise.

  python dft_sp_from_geom.py --manifest <m.parquet> --skip 0 --max 50 \
      --out-csv chunk_00000.csv --workers 48 --maxcore 1500
"""
from __future__ import annotations
import argparse, glob, os, shutil, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # pipeline/compute
import conf_funnel_v3  # noqa: F401  # MUST precede thermo_orca to break the circular import
import thermo_orca as T   # reuse calc_orca_sp / _ORCA_SOLVENT

HARTREE_KCAL = 627.5094740631
HOMO = "data/cross_benzoin/homo_v6"


def build_manifest(repo: Path, out: Path) -> pd.DataFrame:
    """id -> (smiles, ald_xyz, prod_xyz, thermal_ald_Eh, thermal_prod_Eh). Cached."""
    if out.exists():
        return pd.read_parquet(out)
    base = repo / HOMO
    ald = pd.read_csv(base / "aldehydes_all.csv", low_memory=False)
    prod = pd.read_csv(base / "products_all.csv", low_memory=False)
    # thermal = G_xtb - E_el_xtb (Eh), reused RRHO
    ald["thermal"] = ald["G_xtb"] - ald["xtb_energy"]
    prod["thermal"] = prod["G_xtb"] - prod["xtb_energy"]
    # IDs must be int-strings to match the xyz filenames; concat can float-ify them
    # ("2.0" != "2"), so normalise via int.
    def _idstr(s):
        return s.astype(float).astype("Int64").astype(str)
    ald["id"] = _idstr(ald["id"])
    ald_t = ald.set_index("id")["thermal"]
    prod = prod[prod["id"].notna() & prod["donor_id"].notna()].copy()
    prod["id"] = _idstr(prod["id"])
    prod["donor_id"] = _idstr(prod["donor_id"])
    # map id -> saved xyz path (glob once)
    amap = {Path(p).stem.split("_", 1)[1]: p for p in glob.glob(f"{base}/chunk_*/xyz_ald/ald_*.xyz")}
    pmap = {Path(p).stem.split("_", 1)[1]: p for p in glob.glob(f"{base}/chunk_*/xyz_prod/prod_*.xyz")}
    rows = []
    for _, r in prod.iterrows():
        if pd.notna(r.get("error")):
            continue
        i, di = str(r["id"]), str(r["donor_id"])
        if di not in amap or i not in pmap or di not in ald_t.index:
            continue
        rows.append(dict(id=r["id"], smiles=r.get("smiles"), ald_xyz=amap[di], prod_xyz=pmap[i],
                         thermal_ald=float(ald_t[di]), thermal_prod=float(r["thermal"])))
    m = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    print(f"manifest: {len(m)} molecules -> {out}")
    return m


def _sp_nodelocal(src_xyz, method, basis, osolv, maxcore, orca_bin, timeout):
    """Copy the saved geom into node-local scratch, run ORCA SP there, clean up.
    Keeps ALL ORCA scratch off the shared FS (calc_orca_sp writes orca_sp/ next to the xyz).
    The `finally` rmtree frees the per-SP scratch even on timeout/failure (inode safety)."""
    wd = Path(tempfile.mkdtemp(prefix="dftsp_", dir=os.environ.get("TMPDIR", "/tmp")))
    try:
        local = wd / "mol.xyz"
        shutil.copy(src_xyz, local)
        return T.calc_orca_sp(local, method, basis, osolv, maxcore_mb=maxcore,
                              orca_bin=orca_bin, timeout=timeout)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def _one(args):
    row, method, basis, solv, maxcore, orca_bin, timeout = args
    osolv = T._ORCA_SOLVENT.get(solv, "DMSO")
    Ea = _sp_nodelocal(row["ald_xyz"], method, basis, osolv, maxcore, orca_bin, timeout)
    Ep = _sp_nodelocal(row["prod_xyz"], method, basis, osolv, maxcore, orca_bin, timeout)
    out = dict(id=row["id"], smiles=row["smiles"], E_ald_orca_Eh=Ea, E_prod_orca_Eh=Ep, error=None)
    if Ea is None or Ep is None:
        out["error"] = "orca_sp_failed"; out["dG_orca_kcal"] = None
        return out
    G_ald = Ea + row["thermal_ald"]
    G_prod = Ep + row["thermal_prod"]
    out["dG_orca_kcal"] = (G_prod - 2.0 * G_ald) * HARTREE_KCAL
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/scratch-shared/schen3/benzoin-dg")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--method", default="r2SCAN-3c")
    ap.add_argument("--basis", default="def2-mTZVP")
    ap.add_argument("--solvent", default="dmso")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--maxcore", type=int, default=1500)
    ap.add_argument("--timeout", type=int, default=3600, help="per-SP wall cap (s); stuck SPs fail fast")
    ap.add_argument("--orca-bin", default="/home/schen3/orca/orca")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    man = Path(args.manifest) if args.manifest else repo / "data/raw/dft_sp_funnelv3/manifest.parquet"
    m = build_manifest(repo, man)
    sl = m.iloc[args.skip: args.skip + args.max]
    if args.smoke:
        sl = sl.head(3)
    print(f"scoring {len(sl)} molecules (skip={args.skip}) workers={args.workers}")

    tasks = [(r, args.method, args.basis, args.solvent, args.maxcore, args.orca_bin, args.timeout)
             for _, r in sl.iterrows()]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for f in as_completed(futs):
            results.append(f.result())
    out = pd.DataFrame(results)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    ok = out["dG_orca_kcal"].notna().sum()
    print(f"wrote {args.out_csv}  {ok}/{len(out)} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
