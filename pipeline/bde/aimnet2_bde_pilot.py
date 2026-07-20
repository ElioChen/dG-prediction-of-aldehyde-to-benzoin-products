#!/usr/bin/env python
"""AIMNet2-NSE zero-shot BDE pilot (2026-07-16 evening, user question: is the MLIP/MACE
route from BDE_prediction.md sec 6 worth starting now?).

Cheap, low-risk scouting probe -- NOT the Phase-4 production MLIP campaign
BDE_prediction.md recommends sequencing later. Reuses the exact 17 successful
molecules from `dft_bde_geom_arbitration.py` (job 24665669) that already have a real
r2SCAN-3c DFT-optimized-geometry BDE reference, so this is a direct, apples-to-apples
comparison against a ground truth we already paid for -- no new DFT compute needed.

Method: AIMNet2-NSE (open-shell-aware universal MLIP, github.com/isayevlab/aimnetcentral)
single-point energy on (a) the parent aldehyde at its existing GFN2-xTB geometry (no
re-optimization, for speed) and (b) the acyl radical fragment, GFN2-xTB-optimized from a
naive split (reusing calc_bde.py's mol_with_bonds/split_at_bond/_run_xtb_opt -- the acyl
radical geometry from dft_bde_geom_arbitration.py's own run wasn't kept, node-local scratch).
BDE_aimnet = E(acyl) + E(H atom) - E(parent), all in eV (empirically confirmed: H atom
energy -13.86 eV matches -0.5 Hartree, water -2081 eV matches ~-76.5 Hartree), converted
to kcal/mol.

Usage: /gpfs/scratch1/shared/schen3/envs/aimnet2/bin/python aimnet2_bde_pilot.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aimnet.calculators import AIMNet2Calculator

sys.path.insert(0, "/scratch-shared/schen3/benzoin-dg/pipeline/compute")
import ald_descriptors_qm as A  # noqa: E402
from calc_bde import mol_with_bonds, split_at_bond, _xyz_block, _run_xtb_opt  # noqa: E402

EV_TO_KCAL = 23.060548
XTB_BIN = "/home/schen3/xtb/bin/xtb"
SHARD_DIR = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/dft_bde_geom_arbitration")
ALD_ALL = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/aldehydes_all.csv")
WORK = Path("/tmp/aimnet2_bde_pilot")


def aimnet_energy(calc, symbols, coords, mult):
    Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16,
         "Cl": 17, "As": 33, "Se": 34, "Br": 35, "I": 53}
    numbers = torch.tensor([[Z[s] for s in symbols]])
    coord = torch.tensor([coords.tolist()], dtype=torch.float64)
    out = calc({"coord": coord, "numbers": numbers,
                "charge": torch.tensor([0.0]), "mult": torch.tensor([float(mult)])})
    return float(out["energy"].item())


def main() -> int:
    shards = sorted(SHARD_DIR.glob("shard_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    ok = df[df["error"].isna()].copy()
    print(f"{len(ok)}/{len(df)} molecules with a real DFT reference available", flush=True)

    ald_all = pd.read_csv(ALD_ALL, usecols=["id", "xyz_file"], low_memory=False).dropna(subset=["id"])
    ald_all["id_str"] = ald_all["id"].astype(float).astype(int).astype(str)
    ok["id_str"] = ok["id"].astype(float).astype(int).astype(str)
    ok = ok.merge(ald_all[["id_str", "xyz_file"]], on="id_str", how="left")

    calc = AIMNet2Calculator("aimnet2-nse")
    h_e = aimnet_energy(calc, ["H"], np.array([[0.0, 0.0, 0.0]]), mult=2)
    print(f"H atom AIMNet2-NSE energy: {h_e:.4f} eV", flush=True)

    rows = []
    WORK.mkdir(parents=True, exist_ok=True)
    for _, r in ok.iterrows():
        wd = WORK / f"m{r['id_str']}"
        rec = {"id": r["id_str"], "sample_group": r["sample_group"],
               "bde_gxtb_kcal": r["bde_gxtb_kcal"], "bde_dft_opt_dftgeom_kcal": r["bde_dft_opt_dftgeom_kcal"]}
        try:
            xyz_file = r["xyz_file"]
            symbols, coords = A.parse_xyz(Path(xyz_file).read_text())
            hits = A.find_aldehyde_atoms(symbols, coords)
            c_idx, o_idx, _ = hits[0]
            dist = np.linalg.norm(coords - coords[c_idx], axis=1)
            h_idx = next(k for k in range(len(symbols))
                         if symbols[k] == "H" and dist[k] < A.CH_BOND_MAX and k != c_idx)
            mol = mol_with_bonds(xyz_file)
            (symA, coordA), (symB, coordB) = split_at_bond(mol, c_idx, h_idx, coords, symbols)
            if len(symB) != 1:
                (symA, coordA), (symB, coordB) = (symB, coordB), (symA, coordA)
            e_parent = aimnet_energy(calc, symbols, coords, mult=1)
            e_opt = _run_xtb_opt(_xyz_block(symA, coordA), wd / "acyl", XTB_BIN, charge=0, uhf=1)
            if e_opt is None:
                raise RuntimeError("xtb acyl opt failed")
            opt_xyz = (wd / "acyl" / "xtbopt.xyz")
            symO, coordO = A.parse_xyz(opt_xyz.read_text())
            e_acyl = aimnet_energy(calc, symO, coordO, mult=2)
            bde_aimnet = (e_acyl + h_e - e_parent) * EV_TO_KCAL
            rec["bde_aimnet_kcal"] = round(bde_aimnet, 3)
        except Exception as e:
            rec["error"] = str(e)
        rows.append(rec)
        print(rec, flush=True)
        shutil.rmtree(wd, ignore_errors=True)

    out = pd.DataFrame(rows)
    out.to_csv("/tmp/aimnet2_bde_pilot_result.csv", index=False)
    good = out[out.get("bde_aimnet_kcal").notna()] if "bde_aimnet_kcal" in out.columns else out.iloc[0:0]
    print(f"\n{len(good)}/{len(out)} succeeded", flush=True)
    if len(good):
        good = good.assign(
            err_gxtb=(good["bde_gxtb_kcal"] - good["bde_dft_opt_dftgeom_kcal"]).abs(),
            err_aimnet=(good["bde_aimnet_kcal"] - good["bde_dft_opt_dftgeom_kcal"]).abs())
        print(good[["id", "sample_group", "bde_dft_opt_dftgeom_kcal", "bde_gxtb_kcal",
                     "bde_aimnet_kcal", "err_gxtb", "err_aimnet"]].to_string(index=False))
        print(f"\nmean|err| g-xTB vs true DFT:    {good['err_gxtb'].mean():.2f} kcal/mol")
        print(f"mean|err| AIMNet2 vs true DFT:  {good['err_aimnet'].mean():.2f} kcal/mol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
