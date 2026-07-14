#!/usr/bin/env python
"""Cross-benzoin r2SCAN-3c ΔG by SP on SAVED cross_benzoin product geometries.

Cross analogue of dft_sp_from_geom.py (which only handles the homo,
donor==acceptor case: `dG_orca = G_orca(product) - 2*G_orca(aldehyde)`).
Here donor != acceptor, so TWO distinct aldehyde G_orca values are needed:

    dG_orca = G_orca(product) - G_orca(donor) - G_orca(acceptor)

The aldehyde side needs ZERO new DFT compute: the full-library homo DFT-SP
campaign (data/raw/dft_sp_funnelv3/chunk_*.csv, ~219k/220,859 molecules,
r2SCAN-3c on the SAME funnel_v3 geometries used everywhere else in this
project) already has E_ald_orca_Eh for essentially every aldehyde, keyed by
the same numeric `id` (= 0-based row index in aldehydes_clean_v6.csv) used
in aldehydes_all.csv. So only the PRODUCT side -- new cross substrate
combinations that have never been computed before -- needs a new ORCA SP,
one per row of the input products CSV (e.g. cb_featurize.py's output, or the
consolidated cross_pilot_v1_products.csv).

    G_orca(species) = E_orca_SP(species) + (G_xtb(species) - E_el_xtb(species))

reusing the xTB RRHO thermal correction for every species, exactly as in
dft_sp_from_geom.py.

Usage:
  python dft_sp_cross_from_geom.py \
      --products-csv data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_products.csv \
      --skip 0 --max 50 --out-csv chunk_00000.csv --workers 48
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # pipeline/compute
import conf_funnel_v3  # noqa: F401  # MUST precede thermo_orca to break the circular import
import thermo_orca as T   # reuse calc_orca_sp / _ORCA_SOLVENT
from rdkit import Chem

HARTREE_KCAL = 627.5094740631
REPO_DEFAULT = "/scratch-shared/schen3/benzoin-dg"


def build_aldehyde_id_map(clean_v6_csv: Path) -> dict[str, str]:
    """canonical SMILES -> numeric id (0-based row index), the id scheme shared by
    aldehydes_all.csv and data/raw/dft_sp_funnelv3/chunk_*.csv."""
    out: dict[str, str] = {}
    with open(clean_v6_csv, encoding="utf-8-sig", newline="") as fh:
        for idx, row in enumerate(csv.DictReader(fh)):
            smi = (row.get("SMILES") or "").strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                out[Chem.MolToSmiles(mol, canonical=True)] = str(idx)
    return out


def build_aldehyde_thermal(aldehydes_all_csv: Path) -> dict[str, float]:
    """numeric id -> thermal correction (G_xtb - xtb_energy, Eh)."""
    out: dict[str, float] = {}
    with open(aldehydes_all_csv, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            g, e = row.get("G_xtb"), row.get("xtb_energy")
            if not g or not e:
                continue
            # ids are stored as floats ("2.0") in aldehydes_all.csv
            out[str(int(float(row["id"])))] = float(g) - float(e)
    return out


def build_aldehyde_orca_e(dft_sp_dir: Path) -> dict[str, float]:
    """numeric id -> E_ald_orca_Eh from the existing full-library homo DFT-SP campaign."""
    out: dict[str, float] = {}
    for f in glob.glob(str(dft_sp_dir / "chunk_*.csv")):
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                e = row.get("E_ald_orca_Eh")
                if e:
                    out[str(int(float(row["id"])))] = float(e)
    return out


def build_manifest(products_csv: Path, id_map: dict[str, str],
                   thermal: dict[str, float], ald_e: dict[str, float]) -> list[dict]:
    rows = []
    with open(products_csv, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                continue
            xyz = row.get("xyz_file")
            if not xyz or not Path(xyz).exists():
                continue
            g_xtb, e_xtb = row.get("G_xtb"), row.get("xtb_energy")
            if not g_xtb or not e_xtb:
                continue
            try:
                dcanon = Chem.MolToSmiles(Chem.MolFromSmiles(row["donor_smiles"]), canonical=True)
                acanon = Chem.MolToSmiles(Chem.MolFromSmiles(row["acceptor_smiles"]), canonical=True)
                did, aid = id_map[dcanon], id_map[acanon]
                G_donor = ald_e[did] + thermal[did]
                G_acceptor = ald_e[aid] + thermal[aid]
            except (KeyError, TypeError):
                continue   # aldehyde missing from the homo DFT-SP cache -- skip, don't guess
            rows.append(dict(
                id=row["id"], smiles=row["smiles"], prod_xyz=xyz,
                thermal_prod=float(g_xtb) - float(e_xtb),
                G_donor_orca_Eh=G_donor, G_acceptor_orca_Eh=G_acceptor,
            ))
    return rows


def _sp_nodelocal(src_xyz, method, basis, osolv, maxcore, orca_bin, timeout):
    wd = Path(tempfile.mkdtemp(prefix="dftsp_cross_", dir=os.environ.get("TMPDIR", "/tmp")))
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
    Ep = _sp_nodelocal(row["prod_xyz"], method, basis, osolv, maxcore, orca_bin, timeout)
    out = dict(id=row["id"], smiles=row["smiles"], E_prod_orca_Eh=Ep, error=None)
    if Ep is None:
        out["error"] = "orca_sp_failed"; out["dG_orca_kcal"] = None
        return out
    G_prod = Ep + row["thermal_prod"]
    out["dG_orca_kcal"] = (G_prod - row["G_donor_orca_Eh"] - row["G_acceptor_orca_Eh"]) * HARTREE_KCAL
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--products-csv", required=True)
    ap.add_argument("--clean-v6", default=None)
    ap.add_argument("--aldehyde-cache", default=None)
    ap.add_argument("--dft-sp-dir", default=None)
    ap.add_argument("--manifest-cache", default=None,
                    help="cache the built manifest here (CSV); an array of tasks sharing "
                         "one products-csv should point at the SAME cache path so only the "
                         "first task pays for rebuilding the ~220k-row aldehyde lookups")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--method", default="r2SCAN-3c")
    ap.add_argument("--basis", default="def2-mTZVP")
    ap.add_argument("--solvent", default="dmso")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--maxcore", type=int, default=1500)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--orca-bin", default="/home/schen3/orca/orca")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    clean_v6 = Path(args.clean_v6 or repo / "data/library/aldehydes_clean_v6.csv")
    ald_cache = Path(args.aldehyde_cache or repo / "data/cross_benzoin/homo_v6/aldehydes_all.csv")
    dft_sp_dir = Path(args.dft_sp_dir or repo / "data/raw/dft_sp_funnelv3")

    manifest_cache = Path(args.manifest_cache) if args.manifest_cache else None
    if manifest_cache and manifest_cache.exists():
        print(f"loading cached manifest from {manifest_cache}")
        with open(manifest_cache, encoding="utf-8-sig", newline="") as fh:
            m = [{**row, "thermal_prod": float(row["thermal_prod"]),
                  "G_donor_orca_Eh": float(row["G_donor_orca_Eh"]),
                  "G_acceptor_orca_Eh": float(row["G_acceptor_orca_Eh"])}
                 for row in csv.DictReader(fh)]
    else:
        print("building aldehyde id map / thermal / DFT-SP-energy lookups ...")
        id_map = build_aldehyde_id_map(clean_v6)
        thermal = build_aldehyde_thermal(ald_cache)
        ald_e = build_aldehyde_orca_e(dft_sp_dir)
        print(f"  {len(id_map)} aldehyde ids, {len(thermal)} thermal corrections, "
              f"{len(ald_e)} existing DFT-SP energies")
        m = build_manifest(Path(args.products_csv), id_map, thermal, ald_e)
        if manifest_cache:
            # Array tasks can all start before any of them sees the cache file and race to
            # build it concurrently. Write-to-temp + atomic os.replace means whichever task
            # finishes last simply overwrites with an equally-valid, complete file -- never
            # a half-written or interleaved one that a slower task could read mid-write.
            manifest_cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = manifest_cache.with_suffix(f".tmp{os.getpid()}")
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(m[0].keys()) if m else
                                    ["id", "smiles", "prod_xyz", "thermal_prod",
                                     "G_donor_orca_Eh", "G_acceptor_orca_Eh"])
                w.writeheader(); w.writerows(m)
            os.replace(tmp, manifest_cache)
            print(f"cached manifest -> {manifest_cache}")
    print(f"manifest: {len(m)} cross products with a fully cached aldehyde side")
    sl = m[args.skip: args.skip + args.max]
    if args.smoke:
        sl = sl[:3]

    # Resume-safe (per array task / chunk): an ORCA SP on a benzoin-sized product can take
    # tens of minutes, so losing an in-flight chunk's already-completed rows to a wall-clock
    # timeout or preemption would be costly. Skip ids already present in this task's own
    # out-csv and flush each row as it completes instead of collecting in memory and writing
    # once at the end. Use --skip/--max (see slurm/submit_dft_sp_cross_array.sh) to run this
    # as a CHUNK-based array across multiple nodes for large manifests, same pattern as
    # cb_featurize.py / dft_sp_from_geom.py.
    out_path = Path(args.out_csv)
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8-sig", newline="") as fh:
            done_ids = {row["id"] for row in csv.DictReader(fh) if row.get("dG_orca_kcal")}
    sl = [r for r in sl if r["id"] not in done_ids]
    print(f"scoring {len(sl)} products (skip={args.skip}, {len(done_ids)} already done) "
          f"workers={args.workers}")

    fieldnames = ["id", "smiles", "E_prod_orca_Eh", "dG_orca_kcal", "error"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (out_path.exists() and out_path.stat().st_size > 0)
    tasks = [(r, args.method, args.basis, args.solvent, args.maxcore, args.orca_bin, args.timeout)
             for r in sl]
    n_ok = 0
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, t) for t in tasks]
            for f in as_completed(futs):
                r = f.result()
                w.writerow({k: r.get(k) for k in fieldnames}); fh.flush()
                n_ok += r.get("dG_orca_kcal") is not None
    print(f"wrote {out_path}  {n_ok}/{len(tasks)} new ok  "
          f"({len(done_ids) + n_ok}/{len(m)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
