#!/usr/bin/env python
"""Bond dissociation energy (BDE) via real xtb homolysis (not an ML surrogate — see
bde-descriptor-idea memory for why: method-consistent with the rest of this project).

Two bonds, mechanistically tied to the benzoin condensation mechanism:
  - aldehyde: the C(=O)-H bond broken/functionalized during Breslow-intermediate formation.
  - product:  the NEW C-C bond (ketC-carbC) formed by the coupling -- this is literally the
    bond whose formation energy the whole model is trying to predict (dG_orca), so its BDE
    may be an unusually direct descriptor.

Method: RDKit DetermineBonds on the already-optimized funnel_v3 geometry (reused from the
mordred pipeline) to get connectivity -> split at the target bond via GetMolFrags -> xtb
--opt on each fragment (radical, --uhf 1) -> BDE = E(fragA) + E(fragB) - E(parent), all at
GFN2-xTB (same level as everything else in this project). Parent energy is NOT reused from
the funnel_v3 result (that's GFN2-ohess with thermal correction, not the same input file
format) -- recomputed here as a plain --opt single point on the same geometry for a
consistent parent/fragment method.

PILOT usage (tests feasibility on a small sample before any full-library commitment):
  python calc_bde.py --which aldehydes --n 20 --out /tmp/bde_pilot_ald.csv
  python calc_bde.py --which products --n 20 --out /tmp/bde_pilot_prod.csv
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

import ald_descriptors_qm as A
import featurize_product as FP

RDLogger.DisableLog("rdApp.*")
HARTREE_TO_KCAL = 627.509474
H_ATOM_E = {}  # cache: xtb GFN2 energy of a lone H atom is a per-run constant


def _run_xtb_opt(xyz_str: str, work_dir: Path, xtb_bin: str, charge: int, uhf: int, timeout=180):
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "input.xyz").write_text(xyz_str, encoding="utf-8")
    cmd = [xtb_bin, "input.xyz", "--opt", "tight", "--gfn", "2",
          "--charge", str(charge), "--uhf", str(uhf), "--norestart", "--parallel", "1"]
    try:
        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    m = A._parse_xtb_energy(r.stdout)
    return m


def _run_xtb_sp(xyz_str: str, work_dir: Path, xtb_bin: str, charge: int, uhf: int, timeout=120):
    """single point, no opt -- for the trivial 1-atom H fragment (nothing to optimize)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "input.xyz").write_text(xyz_str, encoding="utf-8")
    cmd = [xtb_bin, "input.xyz", "--gfn", "2", "--charge", str(charge), "--uhf", str(uhf),
          "--norestart", "--parallel", "1"]
    try:
        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return A._parse_xtb_energy(r.stdout)


def _xyz_block(symbols, coords) -> str:
    lines = [str(len(symbols)), ""]
    for s, c in zip(symbols, coords):
        lines.append(f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}")
    return "\n".join(lines) + "\n"


def mol_with_bonds(xyz_file: str):
    mol = Chem.MolFromXYZFile(xyz_file)
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None
    return mol


def split_at_bond(mol, i: int, j: int, coords: np.ndarray, symbols: list[str]):
    """Break bond (i,j) [0-based, RDKit atom order == xyz atom order] and return the two
    fragments as (symbols, coords) each, using the ORIGINAL (parent) geometry as the
    starting point for each fragment's own xtb optimization."""
    em = Chem.RWMol(mol)
    if em.GetBondBetweenAtoms(i, j) is None:
        return None
    em.RemoveBond(i, j)
    frags = Chem.GetMolFrags(em, asMols=False, sanitizeFrags=False)
    fa = next(f for f in frags if i in f)
    fb = next(f for f in frags if j in f)
    if fa is fb:  # bond was part of a ring -- not a true bridge, can't split like this
        return None
    out = []
    for frag_idx in (fa, fb):
        sub_sym = [symbols[k] for k in frag_idx]
        sub_coord = coords[list(frag_idx)]
        out.append((sub_sym, sub_coord))
    return out


def bde_aldehyde_ch(xyz_file: str, xtb_bin: str, work_dir: Path) -> dict:
    row = {"bde_ald_CH_kcal": np.nan, "E_parent": np.nan, "E_fragA": np.nan, "E_fragB": np.nan}
    xyz = Path(xyz_file).read_text()
    symbols, coords = A.parse_xyz(xyz)
    hits = A.find_aldehyde_atoms(symbols, coords)
    if not hits:
        return row
    c_idx, o_idx, _ = hits[0]
    dist = np.linalg.norm(coords - coords[c_idx], axis=1)
    h_idx = next((k for k in range(len(symbols))
                 if symbols[k] == "H" and dist[k] < A.CH_BOND_MAX and k != c_idx), None)
    if h_idx is None:
        return row
    mol = mol_with_bonds(xyz_file)
    if mol is None:
        return row
    split = split_at_bond(mol, c_idx, h_idx, coords, symbols)
    if split is None:
        return row
    (symA, coordA), (symB, coordB) = split
    # fragB should be the lone H atom
    if len(symB) != 1:
        (symA, coordA), (symB, coordB) = (symB, coordB), (symA, coordA)
    if len(symB) != 1:
        return row  # neither side is the isolated H -- unexpected topology, skip

    e_parent = _run_xtb_sp(xyz, work_dir / "parent", xtb_bin, charge=0, uhf=0)
    e_fragA = _run_xtb_opt(_xyz_block(symA, coordA), work_dir / "fragA", xtb_bin, charge=0, uhf=1)
    if "H" not in H_ATOM_E:
        H_ATOM_E["H"] = _run_xtb_sp(_xyz_block(symB, coordB), work_dir / "fragB", xtb_bin, charge=0, uhf=1)
    e_fragB = H_ATOM_E["H"]
    row.update(E_parent=e_parent, E_fragA=e_fragA, E_fragB=e_fragB)
    if None not in (e_parent, e_fragA, e_fragB):
        row["bde_ald_CH_kcal"] = round((e_fragA + e_fragB - e_parent) * HARTREE_TO_KCAL, 3)
    return row


def bde_product_cc(xyz_file: str, xtb_bin: str, work_dir: Path) -> dict:
    row = {"bde_prod_CC_kcal": np.nan, "E_parent": np.nan, "E_fragA": np.nan, "E_fragB": np.nan}
    xyz = Path(xyz_file).read_text()
    symbols, coords = A.parse_xyz(xyz)
    core = FP.find_benzoin_core(symbols, coords)
    if core is None:
        return row
    i, j = core["ketC"], core["carbC"]
    mol = mol_with_bonds(xyz_file)
    if mol is None:
        return row
    split = split_at_bond(mol, i, j, coords, symbols)
    if split is None:
        return row
    (symA, coordA), (symB, coordB) = split

    e_parent = _run_xtb_sp(xyz, work_dir / "parent", xtb_bin, charge=0, uhf=0)
    e_fragA = _run_xtb_opt(_xyz_block(symA, coordA), work_dir / "fragA", xtb_bin, charge=0, uhf=1)
    e_fragB = _run_xtb_opt(_xyz_block(symB, coordB), work_dir / "fragB", xtb_bin, charge=0, uhf=1)
    row.update(E_parent=e_parent, E_fragA=e_fragA, E_fragB=e_fragB)
    if None not in (e_parent, e_fragA, e_fragB):
        row["bde_prod_CC_kcal"] = round((e_fragA + e_fragB - e_parent) * HARTREE_TO_KCAL, 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--n", type=int, default=None, help="pilot mode: first N molecules")
    ap.add_argument("--chunk-id", type=int, default=None, help="array mode: which chunk")
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--out", default=None, help="pilot mode: single output CSV")
    ap.add_argument("--out-dir", default=None, help="array mode: chunk_NNNN.csv written here")
    ap.add_argument("--xtb-bin", default="/home/schen3/xtb/bin/xtb")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
    src = H / f"{args.which}_all.csv"
    df = pd.read_csv(src, usecols=["id", "xyz_file", "error"], dtype=str,
                     keep_default_na=False, low_memory=False)
    df = df[(df["error"] == "") & (df["xyz_file"] != "")].drop_duplicates("id").reset_index(drop=True)

    if args.chunk_id is not None:
        lo, hi = args.chunk_id * args.chunk_size, min((args.chunk_id + 1) * args.chunk_size, len(df))
        if lo >= len(df):
            print(f"chunk {args.chunk_id}: out of range, nothing to do", flush=True); return
        df = df.iloc[lo:hi].reset_index(drop=True)
        out_path = Path(args.out_dir) / f"chunk_{args.chunk_id:04d}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            existing = pd.read_csv(out_path, usecols=["id"])
            if len(existing) >= len(df):
                print(f"chunk {args.chunk_id}: already done ({len(existing)}/{len(df)}) -- skip", flush=True)
                return
        print(f"{args.which} chunk {args.chunk_id}: rows {lo}:{hi} ({len(df)})", flush=True)
    else:
        df = df.head(args.n or 20)
        out_path = Path(args.out)
        print(f"piloting BDE on {len(df)} {args.which}", flush=True)

    wd_root = Path(args.work_dir or "/tmp/bde_pilot")
    rows = []
    for _, rec in df.iterrows():
        wd = wd_root / f"m{rec['id']}"
        try:
            if args.which == "aldehydes":
                r = bde_aldehyde_ch(rec["xyz_file"], args.xtb_bin, wd)
            else:
                r = bde_product_cc(rec["xyz_file"], args.xtb_bin, wd)
        except Exception as e:
            r = {"error": str(e)}
        r["id"] = rec["id"]
        rows.append(r)
        print(rec["id"], r, flush=True)
        shutil.rmtree(wd, ignore_errors=True)

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    key = "bde_ald_CH_kcal" if args.which == "aldehydes" else "bde_prod_CC_kcal"
    ok = result[key].notna().sum() if key in result.columns else 0
    print(f"wrote {out_path} ({ok}/{len(result)} succeeded)", flush=True)


if __name__ == "__main__":
    main()
