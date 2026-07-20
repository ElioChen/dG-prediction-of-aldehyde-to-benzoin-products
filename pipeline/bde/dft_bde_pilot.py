#!/usr/bin/env python
"""DFT ground-truth arbitration pilot for the aldehyde formyl C-H bond (2026-07-15).

Motivation: B0's diagnosis ("ALFABET zero-shot barely correlates with this project's own
g-xTB BDE for aldehydes, r=0.12") assumed the g-xTB label is the reliable reference and
ALFABET is the one that's "wrong" (different oracle, M06-2X vs g-xTB). That assumption was
never independently checked -- it's equally possible g-xTB's BDE is itself noisy on this
bond and ALFABET's tight 85-94 kcal/mol range is closer to reality. This script computes a
genuine third reference (DFT r2SCAN-3c, this project's own production DFT level) for a
small, deliberately BDE-range-spanning sample, to see which of {g-xTB, ALFABET} tracks it
better.

Electronic BDE only (E, not G/BDFE) -- matches the project's own SHAP finding that BDE
carries the signal and BDFE's extra RRHO thermal correction is ~noise (see
bde-descriptor-idea memory / METHODS_BDE_gxtb docs), so skipping fragment Hessians here
is a deliberate cost-saving choice, not a shortcut that loses information.

Cost-saving reuse: the PARENT aldehyde's DFT electronic energy (E_ald_orca_Eh) is already
computed for ~219k/220,859 molecules in data/raw/dft_sp_funnelv3/chunk_*.csv (the existing
production dG-model DFT-SP campaign) -- this script only computes NEW DFT single points for
the two radical FRAGMENTS (formyl C-H homolysis), cutting compute by ~2/3 vs computing all
three species fresh.

Method: reuses this project's established recipe end-to-end --
  1. mol_with_bonds/split_at_bond (pipeline/compute/calc_bde.py) to split the existing
     xtb-optimized parent geometry at the formyl C-H bond.
  2. _run_xtb_opt (calc_bde.py) -- GFN2 --opt tight, uhf=1 -- to relax each radical
     fragment's geometry (same "optimize fragment cheaply at xtb, DFT-SP the result"
     pattern calc_bde_gxtb_product_cross.py uses for g-xTB).
  3. A NEW open-shell (multiplicity=2, UKS) ORCA r2SCAN-3c/CPCM(DMSO) single point on each
     optimized fragment -- thermo_orca.calc_orca_sp hardcodes closed-shell "0 1" in its
     ORCA input, so this script has its own small input builder for the doublet case
     rather than modifying that shared module (also: a DFT-SP array job from a different,
     concurrent session was observed actively running against this same ORCA
     installation during this work -- editing thermo_orca.py in place was avoided as an
     unnecessary risk to whatever that job depends on).

Usage:
  python dft_bde_pilot.py --n 25 --out /tmp/dft_bde_pilot.csv

Scale-up (2026-07-15, Phase-1 next-step #3): the n=25 pilot (24/25 succeeded) showed g-xTB
tracks DFT much better than ALFABET (r=0.63 vs 0.44 -- see memory `dft-arbitration-result`),
but n=24 is too small to pin down whether g-xTB's own MAE-vs-DFT gap (~14 kcal/mol) is a
real noise floor that already caps B4/B5's 2-3.5 kcal MAE, or just sampling noise. Since
each molecule costs ~7 CPU-min (2 xtb opts + 2 ORCA SPs, sequential), scaling to n=100 on
one CPU would take ~11h -- `--n-shards`/`--shard-idx` split the SAME deliberately
BDE-range-spanning sample (computed once from `--n`, deterministically) into independent
chunks for a SLURM array, one process per shard, each writing its own `--out` file; combine
with pipeline/bde/merge_dft_bde_pilot_shards.py afterward.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compute"))
import ald_descriptors_qm as A  # noqa: E402
from calc_bde import _run_xtb_opt, mol_with_bonds, split_at_bond, _xyz_block  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc import qc_filter  # noqa: E402

RDLogger.DisableLog("rdApp.*")
HARTREE_TO_KCAL = 627.509474
H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
REPO = Path("/scratch-shared/schen3/benzoin-dg")
XTB_BIN = "/home/schen3/xtb/bin/xtb"
ORCA_BIN = "/home/schen3/orca/orca"


def load_dft_ald_energies() -> pd.Series:
    """id -> E_ald_orca_Eh, from the existing full-library DFT-SP campaign (no new compute)."""
    frames = []
    for f in sorted((REPO / "data/raw/dft_sp_funnelv3").glob("chunk_*.csv")):
        df = pd.read_csv(f, usecols=["id", "E_ald_orca_Eh", "error"], dtype=str,
                          keep_default_na=False)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full[(full["error"] == "") & (full["E_ald_orca_Eh"] != "")].drop_duplicates("id")
    full["E_ald_orca_Eh"] = full["E_ald_orca_Eh"].astype(float)
    return full.set_index("id")["E_ald_orca_Eh"]


def orca_sp_open_shell(xyz_path: Path, mult: int, nprocs: int = 1) -> float | None:
    # nprocs=1: matches this project's own convention (e.g. submit_dft_sp_relabel158.sh
    # always passes --orca-nprocs 1) -- ORCA's `%pal nprocs N` needs mpirun, which isn't
    # on PATH in this shell without `module load 2023`; simplest to just not need it.
    """r2SCAN-3c/CPCM(DMSO) SP at the given spin multiplicity (UKS for mult>1) --
    thermo_orca.calc_orca_sp hardcodes closed-shell '0 1', this is its open-shell twin."""
    work_dir = xyz_path.parent / "orca_sp_os"
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(xyz_path, work_dir / "mol.xyz")
    ref = "UKS" if mult > 1 else "RKS"
    pal = f"%pal nprocs {nprocs} end\n\n" if nprocs > 1 else ""
    inp = (f"! r2SCAN-3c {ref} TightSCF NoMOPrint CPCM(DMSO)\n\n"
           f"{pal}%maxcore 1792\n\n"
           f"* xyzfile 0 {mult} mol.xyz\n")
    (work_dir / "input.inp").write_text(inp, encoding="utf-8")
    env = {**os.environ,
           "PATH": f"{Path(ORCA_BIN).parent}:{os.environ.get('PATH', '')}",
           "LD_LIBRARY_PATH": f"{Path(ORCA_BIN).parent / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    try:
        r = subprocess.run([ORCA_BIN, "input.inp"], cwd=str(work_dir), capture_output=True,
                            text=True, timeout=3600, env=env)
    except subprocess.TimeoutExpired:
        return None
    output = r.stdout + r.stderr
    (work_dir / "input.out").write_text(output, encoding="utf-8")
    E = None
    for line in output.splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            try:
                E = float(line.split()[-1])
            except ValueError:
                pass
    for pattern in ["*.gbw", "*.tmp", "*.densities", "*.ges", "*_ges"]:
        for f in work_dir.glob(pattern):
            try:
                f.unlink()
            except Exception:
                pass
    return E


def fragment_dft_bde(row, work_root: Path) -> dict:
    xyz_file = row["xyz_file"]
    if not Path(xyz_file).exists():
        return {"error": "xyz_missing"}
    symbols, coords = A.parse_xyz(Path(xyz_file).read_text())
    hits = A.find_aldehyde_atoms(symbols, coords)
    if not hits:
        return {"error": "no_aldehyde_motif"}
    c_idx, o_idx, _ = hits[0]
    dist = np.linalg.norm(coords - coords[c_idx], axis=1)
    h_idx = next((k for k in range(len(symbols))
                  if symbols[k] == "H" and dist[k] < A.CH_BOND_MAX and k != c_idx), None)
    if h_idx is None:
        return {"error": "no_formyl_h"}
    mol = mol_with_bonds(xyz_file)
    if mol is None:
        return {"error": "determine_bonds_failed"}
    split = split_at_bond(mol, c_idx, h_idx, coords, symbols)
    if split is None:
        return {"error": "split_failed"}
    (symA, coordA), (symB, coordB) = split
    if len(symB) != 1:  # keep fragB as the lone H
        (symA, coordA), (symB, coordB) = (symB, coordB), (symA, coordA)
    if len(symB) != 1:
        return {"error": "fragment_sizes_unexpected"}

    wd = work_root / f"m{row['id']}"
    E_frags = {}
    for tag, sym, coord in (("acyl", symA, coordA), ("h", symB, coordB)):
        fwd = wd / tag
        if len(sym) == 1:  # isolated H: nothing to xtb-optimize
            fwd.mkdir(parents=True, exist_ok=True)
            (fwd / "opt.xyz").write_text(_xyz_block(sym, coord))
        else:
            _run_xtb_opt(_xyz_block(sym, coord), fwd, XTB_BIN, charge=0, uhf=1)
            optxyz = fwd / "xtbopt.xyz"
            if not optxyz.exists():
                return {"error": f"xtb_opt_failed_{tag}"}
            shutil.copy(optxyz, fwd / "opt.xyz")
        E = orca_sp_open_shell(fwd / "opt.xyz", mult=2)
        if E is None:
            return {"error": f"orca_failed_{tag}"}
        E_frags[tag] = E
    shutil.rmtree(wd, ignore_errors=True)
    return {"E_acyl_orca_Eh": E_frags["acyl"], "E_h_orca_Eh": E_frags["h"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default="/scratch-local/dft_bde_pilot")
    ap.add_argument("--n-shards", type=int, default=1,
                     help="split the size-n sample into this many chunks for a SLURM array")
    ap.add_argument("--shard-idx", type=int, default=0,
                     help="which chunk this process computes (0-indexed, < n-shards)")
    args = ap.parse_args()
    assert 0 <= args.shard_idx < args.n_shards

    dft_ald = load_dft_ald_energies()
    print(f"DFT parent energies available for {len(dft_ald)} aldehydes (reused, no new compute)",
          flush=True)

    ald_all = pd.read_csv(H / "aldehydes_all.csv", usecols=["id", "smiles", "xyz_file", "error"],
                           dtype=str, keep_default_na=False)
    ald_all = ald_all[(ald_all["error"] == "") & (ald_all["xyz_file"] != "")]
    labels = pd.read_csv(H / "aldehydes_bdfe_gxtb_descriptors.csv", dtype={"id": str}) \
        .dropna(subset=["bde_gxtb_kcal"])
    labels = labels[qc_filter(labels["bde_gxtb_kcal"])]
    alfabet = pd.read_csv(REPO / "data/cross_benzoin/homo_v6/aldehydes_bde_alfabet.csv",
                           dtype={"id": str})

    df = ald_all.merge(labels, on="id", how="inner").merge(
        alfabet[["id", "bde_alfabet_kcal"]], on="id", how="inner")
    df["id_int"] = df["id"].str.replace(".0", "", regex=False)
    df = df[df["id_int"].isin(dft_ald.index)]
    print(f"{len(df)} aldehydes have DFT parent energy + g-xTB BDE + ALFABET BDE all available",
          flush=True)

    # Deliberately span the g-xTB BDE range (deciles), not a plain random sample -- a
    # DFT arbitration is only informative if it covers low/mid/high BDE, not just the mode.
    df = df.sort_values("bde_gxtb_kcal").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, args.n).round().astype(int)
    sample = df.iloc[idx].drop_duplicates("id").reset_index(drop=True)
    print(f"sampled {len(sample)} aldehydes spanning bde_gxtb_kcal "
          f"[{sample['bde_gxtb_kcal'].min():.1f}, {sample['bde_gxtb_kcal'].max():.1f}]", flush=True)
    if args.n_shards > 1:
        sample = sample.iloc[args.shard_idx::args.n_shards].reset_index(drop=True)
        print(f"shard {args.shard_idx}/{args.n_shards}: {len(sample)} molecules", flush=True)

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in sample.iterrows():
        r = fragment_dft_bde(row, work_root)
        r["id"] = row["id"]
        r["bde_gxtb_kcal"] = row["bde_gxtb_kcal"]
        r["bde_alfabet_kcal"] = row["bde_alfabet_kcal"]
        if "E_acyl_orca_Eh" in r:
            E_parent = dft_ald.loc[row["id_int"]]
            r["bde_dft_kcal"] = (r["E_acyl_orca_Eh"] + r["E_h_orca_Eh"] - E_parent) * HARTREE_TO_KCAL
        rows.append(r)
        print(row["id"], r, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    ok = out["bde_dft_kcal"].notna().sum() if "bde_dft_kcal" in out else 0
    print(f"wrote {args.out}  {ok}/{len(out)} succeeded", flush=True)


if __name__ == "__main__":
    main()
