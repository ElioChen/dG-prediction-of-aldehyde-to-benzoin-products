#!/usr/bin/env python
"""DFT geometry-vs-single-point decomposition of the g-xTB BDE label error (2026-07-16).

Motivation (user question): the n=100 DFT arbitration (`dft_bde_pilot.py`) established a
~14.7 kcal/mol gap between the g-xTB BDE label and an r2SCAN-3c DFT reference -- but that
reference is a DFT SINGLE POINT on the GFN2-xTB geometry (parent from dft_sp_funnelv3, acyl
fragment GFN2-opt then DFT-SP). So it isolates ONLY the single-point electronic-structure
method difference at a FIXED GFN2 geometry. It does NOT tell us whether the GFN2 geometry
itself is bad relative to a DFT-optimized geometry. The label error could be:
  (a) g-xTB single-point energy being worse than DFT   (SP-method component), or
  (b) GFN2-xTB geometry optimization being unfavorable  (geometry component).
This script adds the missing third leg -- a full r2SCAN-3c GEOMETRY OPTIMIZATION of parent
and acyl radical -- so the label's deviation from a DFT-opt "truth" can be split:

  Delta_SP   = bde_gxtb            - bde_dft_sp_gfn2geom   (g-xTB SP error @ fixed GFN2 geom)
  Delta_geom = bde_dft_sp_gfn2geom - bde_dft_opt_dftgeom   (GFN2 vs DFT-opt geometry, at DFT SP)
  Delta_tot  = bde_gxtb            - bde_dft_opt_dftgeom   = Delta_SP + Delta_geom

If |Delta_geom| is small, the label error is dominantly an SP-method issue (more model
capacity won't help; the label is just a different level of theory). If |Delta_geom| is
large, GFN2 geometry is a real culprit and a geometry upgrade would move the labels.

Sampling: deliberately focused on the functional groups the B6 ensemble's residuals
concentrate on (sulfonyl / imine / nitro / N-oxide / phosphorus -- see
PROGRESS_20260714.md 〇-3 sec 3), plus a few "easy" control molecules (plain halogen/ether,
low residual), so the decomposition is measured exactly where the model struggles. Small n
(default ~16) because DFT geometry optimization is ~1-2 orders of magnitude costlier than
the single points the earlier pilots used.

Energy legs, per sampled aldehyde (formyl C-H homolysis, electronic BDE only, no RRHO --
same scope choice as dft_bde_pilot.py):
  parent  SP @GFN2geom : REUSED from dft_sp_funnelv3 (E_ald_orca_Eh), no recompute.
  acyl    SP @GFN2geom : NEW DFT SP on the GFN2-opt acyl fragment (== dft_bde_pilot.py leg).
  parent  OPT (DFT)    : NEW r2SCAN-3c Opt starting from the xtb parent geometry.
  acyl    OPT (DFT)    : NEW r2SCAN-3c UKS Opt (doublet) starting from the GFN2-opt acyl geom.
  H atom               : NEW DFT SP of a lone H (doublet), geometry-independent, computed once.
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

# functional groups the B6 residuals concentrate on (targets) + easy controls
TARGET_GROUPS = [
    ("sulfonyl", "[#16X4](=[OX1])(=[OX1])"),
    ("imine", "[CX3]=[NX2]"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("n_oxide", "[#7+][#8X1-]"),
    ("phosphorus", "[#15]"),
]
CONTROL_GROUPS = [
    ("halogen_plain", "[F,Cl,Br,I]"),
]


def load_dft_ald_energies() -> pd.Series:
    frames = []
    for f in sorted((REPO / "data/raw/dft_sp_funnelv3").glob("chunk_*.csv")):
        df = pd.read_csv(f, usecols=["id", "E_ald_orca_Eh", "error"], dtype=str,
                          keep_default_na=False)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full[(full["error"] == "") & (full["E_ald_orca_Eh"] != "")].drop_duplicates("id")
    full["E_ald_orca_Eh"] = full["E_ald_orca_Eh"].astype(float)
    return full.set_index("id")["E_ald_orca_Eh"]


def _run_orca(work_dir: Path, simple_line: str, charge: int, mult: int,
              nprocs: int, timeout: int) -> float | None:
    """Run an ORCA job (Opt or SP depending on simple_line), return FINAL SINGLE POINT
    ENERGY (for Opt this is the energy at the converged geometry)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pal = f"%pal nprocs {nprocs} end\n\n" if nprocs > 1 else ""
    inp = (f"{simple_line}\n\n{pal}%maxcore 1792\n\n"
           f"* xyzfile {charge} {mult} mol.xyz\n")
    (work_dir / "input.inp").write_text(inp, encoding="utf-8")
    env = {**os.environ,
           "PATH": f"{Path(ORCA_BIN).parent}:{os.environ.get('PATH', '')}",
           "LD_LIBRARY_PATH": f"{Path(ORCA_BIN).parent / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    try:
        r = subprocess.run([ORCA_BIN, "input.inp"], cwd=str(work_dir), capture_output=True,
                            text=True, timeout=timeout, env=env)
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
    for pattern in ["*.gbw", "*.tmp", "*.densities", "*.ges", "*_ges", "*.opt", "*.bibtex"]:
        for f in work_dir.glob(pattern):
            try:
                f.unlink()
            except Exception:
                pass
    return E


def compute_legs(row, work_root: Path, E_parent_sp: float, E_h: float,
                 nprocs: int, timeout: int) -> dict:
    """All energy legs for one aldehyde. E_parent_sp (DFT SP @GFN2geom, reused) and E_h
    (DFT H atom) are passed in. Returns the two new fragment SPs + two DFT-opt energies."""
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
    if len(symB) != 1:
        (symA, coordA), (symB, coordB) = (symB, coordB), (symA, coordA)
    if len(symB) != 1:
        return {"error": "fragment_sizes_unexpected"}

    wd = work_root / f"m{row['id']}"
    out = {}

    # acyl: GFN2-opt (for the SP@GFN2geom leg), then DFT SP on it, then DFT Opt from it.
    acyl_wd = wd / "acyl"
    _run_xtb_opt(_xyz_block(symA, coordA), acyl_wd, XTB_BIN, charge=0, uhf=1)
    optxyz = acyl_wd / "xtbopt.xyz"
    if not optxyz.exists():
        return {"error": "xtb_opt_failed_acyl"}
    shutil.copy(optxyz, acyl_wd / "mol.xyz")
    (acyl_wd / "sp").mkdir(parents=True, exist_ok=True)
    shutil.copy(acyl_wd / "mol.xyz", acyl_wd / "sp" / "mol.xyz")
    E_acyl_sp = _run_orca(acyl_wd / "sp", "! r2SCAN-3c UKS TightSCF NoMOPrint CPCM(DMSO)",
                          charge=0, mult=2, nprocs=nprocs, timeout=timeout)
    if E_acyl_sp is None:
        return {"error": "orca_sp_acyl_failed"}
    out["E_acyl_sp_gfn2geom_Eh"] = E_acyl_sp
    (acyl_wd / "optdir").mkdir(parents=True, exist_ok=True)
    shutil.copy(acyl_wd / "mol.xyz", acyl_wd / "optdir" / "mol.xyz")
    E_acyl_opt = _run_orca(acyl_wd / "optdir",
                           "! r2SCAN-3c UKS Opt TightSCF NoMOPrint CPCM(DMSO)",
                           charge=0, mult=2, nprocs=nprocs, timeout=timeout)
    if E_acyl_opt is None:
        return {"error": "orca_opt_acyl_failed", **out}
    out["E_acyl_opt_dftgeom_Eh"] = E_acyl_opt

    # parent: DFT Opt starting from the existing xtb parent geometry.
    par_wd = wd / "parent" / "optdir"
    par_wd.mkdir(parents=True, exist_ok=True)
    (par_wd / "mol.xyz").write_text(_xyz_block(symbols, coords))
    E_par_opt = _run_orca(par_wd, "! r2SCAN-3c Opt TightSCF NoMOPrint CPCM(DMSO)",
                          charge=0, mult=1, nprocs=nprocs, timeout=timeout)
    if E_par_opt is None:
        return {"error": "orca_opt_parent_failed", **out}
    out["E_parent_opt_dftgeom_Eh"] = E_par_opt

    shutil.rmtree(wd, ignore_errors=True)

    out["bde_dft_sp_gfn2geom_kcal"] = (E_acyl_sp + E_h - E_parent_sp) * HARTREE_TO_KCAL
    out["bde_dft_opt_dftgeom_kcal"] = (E_acyl_opt + E_h - E_par_opt) * HARTREE_TO_KCAL
    return out


def h_atom_energy(work_root: Path, nprocs: int, timeout: int) -> float | None:
    wd = work_root / "h_atom"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "mol.xyz").write_text("1\nH atom\nH 0.0 0.0 0.0\n")
    return _run_orca(wd, "! r2SCAN-3c UKS TightSCF NoMOPrint CPCM(DMSO)",
                     charge=0, mult=2, nprocs=nprocs, timeout=timeout)


def tag_first_group(smi, pats):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    for lab, p in pats:
        if p is not None and m.HasSubstructMatch(p):
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=3,
                    help="molecules per target functional group (BDE-range-spread within group)")
    ap.add_argument("--n-control", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nprocs", type=int, default=1,
                    help="ORCA %%pal nprocs -- keep at 1: mpirun isn't on PATH here "
                         "(same constraint dft_bde_pilot.py already worked around)")
    ap.add_argument("--timeout", type=int, default=21600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default="/scratch-local/dft_bde_geom")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--candidates-csv", default=None,
                     help="if set, skip internal TARGET_GROUPS/per-group/n-control sampling "
                          "and use this pre-selected candidate list instead (must have "
                          "id, smiles, sample_group, bde_gxtb_kcal columns -- e.g. the output "
                          "of select_dft_arbitration_batch.py). Still restricted to molecules "
                          "with a cached parent DFT single-point energy (dft_sp_funnelv3).")
    args = ap.parse_args()
    assert 0 <= args.shard_idx < args.n_shards

    dft_ald = load_dft_ald_energies()
    ald_all = pd.read_csv(H / "aldehydes_all.csv", usecols=["id", "smiles", "xyz_file", "error"],
                           dtype=str, keep_default_na=False)
    ald_all = ald_all[(ald_all["error"] == "") & (ald_all["xyz_file"] != "")]
    alfabet = pd.read_csv(H / "aldehydes_bde_alfabet.csv", dtype={"id": str})

    if args.candidates_csv:
        cand = pd.read_csv(args.candidates_csv, dtype={"id": str})
        cand["id_int"] = cand["id"].str.replace(".0", "", regex=False)
        # candidates carry their own smiles/bde_gxtb_kcal already, but need xyz_file
        # (from ald_all) and bde_alfabet_kcal (for parity with the original pilot's columns)
        sample = cand.merge(ald_all[["id", "xyz_file"]].assign(
            id_int=lambda d: d["id"].str.replace(".0", "", regex=False))[["id_int", "xyz_file"]],
            on="id_int", how="left")
        sample = sample.merge(alfabet.rename(columns={"id": "id_int"})[["id_int", "bde_alfabet_kcal"]],
                               on="id_int", how="left")
        n_before = len(sample)
        sample = sample[sample["id_int"].isin(dft_ald.index) & sample["xyz_file"].notna()].reset_index(drop=True)
        print(f"--candidates-csv: {n_before} candidates, {len(sample)} usable "
              f"(have cached parent DFT-SP energy + xyz_file)", flush=True)
    else:
        labels = pd.read_csv(H / "aldehydes_bdfe_gxtb_descriptors.csv", dtype={"id": str}) \
            .dropna(subset=["bde_gxtb_kcal"])
        labels = labels[qc_filter(labels["bde_gxtb_kcal"])]
        df = ald_all.merge(labels, on="id", how="inner").merge(
            alfabet[["id", "bde_alfabet_kcal"]], on="id", how="inner")
        df["id_int"] = df["id"].str.replace(".0", "", regex=False)
        df = df[df["id_int"].isin(dft_ald.index)].reset_index(drop=True)

        pats_t = [(lab, Chem.MolFromSmarts(sm)) for lab, sm in TARGET_GROUPS]
        pats_c = [(lab, Chem.MolFromSmarts(sm)) for lab, sm in CONTROL_GROUPS]
        df["grp"] = df["smiles"].map(lambda s: tag_first_group(s, pats_t))

        picks = []
        for lab, _ in TARGET_GROUPS:
            sub = df[df["grp"] == lab].sort_values("bde_gxtb_kcal").reset_index(drop=True)
            if len(sub) == 0:
                print(f"WARN: no molecules for target group {lab}", flush=True)
                continue
            idx = np.linspace(0, len(sub) - 1, min(args.per_group, len(sub))).round().astype(int)
            picks.append(sub.iloc[idx].assign(sample_group=lab))
        # controls: molecules matching a plain halogen but NONE of the target groups
        ctrl = df[df["grp"].isna()].copy()
        ctrl = ctrl[ctrl["smiles"].map(lambda s: tag_first_group(s, pats_c) is not None)]
        ctrl = ctrl.sort_values("bde_gxtb_kcal").reset_index(drop=True)
        if len(ctrl):
            idx = np.linspace(0, len(ctrl) - 1, min(args.n_control, len(ctrl))).round().astype(int)
            picks.append(ctrl.iloc[idx].assign(sample_group="control"))
        sample = pd.concat(picks, ignore_index=True).drop_duplicates("id").reset_index(drop=True)

    print(f"selected {len(sample)} molecules: "
          f"{sample['sample_group'].value_counts().to_dict()}", flush=True)
    if args.n_shards > 1:
        sample = sample.iloc[args.shard_idx::args.n_shards].reset_index(drop=True)
        print(f"shard {args.shard_idx}/{args.n_shards}: {len(sample)} molecules", flush=True)

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    E_h = h_atom_energy(work_root, args.nprocs, args.timeout)
    print(f"H-atom DFT energy: {E_h}", flush=True)

    rows = []
    for _, row in sample.iterrows():
        E_parent_sp = float(dft_ald.loc[row["id_int"]])
        r = compute_legs(row, work_root, E_parent_sp, E_h, args.nprocs, args.timeout)
        r["id"] = row["id"]
        r["sample_group"] = row["sample_group"]
        r["smiles"] = row["smiles"]
        r["bde_gxtb_kcal"] = row["bde_gxtb_kcal"]
        r["bde_alfabet_kcal"] = row["bde_alfabet_kcal"]
        r["E_parent_sp_gfn2geom_Eh"] = E_parent_sp
        r["E_h_Eh"] = E_h
        if "bde_dft_opt_dftgeom_kcal" in r:
            r["Delta_SP"] = r["bde_gxtb_kcal"] - r["bde_dft_sp_gfn2geom_kcal"]
            r["Delta_geom"] = r["bde_dft_sp_gfn2geom_kcal"] - r["bde_dft_opt_dftgeom_kcal"]
            r["Delta_tot"] = r["bde_gxtb_kcal"] - r["bde_dft_opt_dftgeom_kcal"]
        rows.append(r)
        print(row["id"], row["sample_group"],
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
               if k in ("bde_gxtb_kcal", "bde_dft_sp_gfn2geom_kcal", "bde_dft_opt_dftgeom_kcal",
                        "Delta_SP", "Delta_geom", "error")}, flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)  # flush incrementally

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    ok = out["bde_dft_opt_dftgeom_kcal"].notna().sum() if "bde_dft_opt_dftgeom_kcal" in out else 0
    print(f"wrote {args.out}  {ok}/{len(out)} full-decomposition succeeded", flush=True)


if __name__ == "__main__":
    main()
