#!/usr/bin/env python3
"""
Re-sample a CLEAN conformer-test set (replaces the dirty subset_conftest.csv).

Why a new set: the old subset_conftest.csv was built before the v2 filters and the
aromatic-only pivot, so it is contaminated with now-out-of-scope molecules (the very
first row is an SF5 reactive_group, several are aliphatic/exotic). A conformer-method
benchmark (RDKit-funnel v1/v2 vs CREST gfnff/gfn2) must run on molecules that are
actually in production scope, otherwise the failures we measure are artefacts of
substrates we will never label.

This set is purpose-built to STRESS the conformer search:
  • in scope      — aromatic_carbo + aromatic_hetero only, all classify()=='keep'
  • flexibility-  — stratified by the rotatable-bond count of the BENZOIN PRODUCT
    stratified     (what we actually conformer-search), since flexibility is the
                   axis along which search stochasticity bites. Bins mirror the
                   funnel's nconf tiers: rigid 0-3 / 4-7 / floppy 8-12 / very 13+.
  • balanced      — equal carbo/hetero, equal per flex bin
  • diverse       — MaxMin (Morgan r2) pick within each (class x bin) cell

Output: data/library/subset_conftest_v2.csv with index, SMILES, cho_class,
product_rotbonds, flex_bin, sel_method.

Usage
  python pipeline/sample_conftest_v2.py --per-cell 5 --cand-per-class 6000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline" / "compute"))
import thermo_orca as Th          # noqa: E402  (_make_benzoin_smiles, _mol_rotbonds)
from filter_smiles import classify  # noqa: E402

FLEX_BINS = [(0, 3, "rigid_0-3"), (4, 7, "mid_4-7"),
             (8, 12, "floppy_8-12"), (13, 999, "very_13+")]
CLASSES = ["aromatic_carbo", "aromatic_hetero"]


def flex_bin(rb: int) -> str:
    for lo, hi, name in FLEX_BINS:
        if lo <= rb <= hi:
            return name
    return "very_13+"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(REPO / "data/library/pool_categorized.csv"),
                    help="clean+classified pool (from select_aromatic)")
    ap.add_argument("--per-cell", type=int, default=5,
                    help="molecules per (class x flex-bin) cell -> total = per_cell*2*4")
    ap.add_argument("--cand-per-class", type=int, default=6000,
                    help="random candidates per class to build products + bin (cost cap)")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--out", default=str(REPO / "data/library/subset_conftest_v2.csv"))
    ap.add_argument("--seed", type=int, default=20260614)
    args = ap.parse_args()

    t0 = time.time()
    pool = pd.read_csv(args.pool)
    pool = pool[pool["cho_class"].isin(CLASSES)].reset_index(drop=True)
    print(f"Pool (aromatic carbo+hetero): {len(pool):,}")

    rng = np.random.default_rng(args.seed)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius, fpSize=args.n_bits)

    # Build a candidate frame: random sample per class, compute the benzoin product,
    # its rotatable-bond count, flex bin, and fingerprint. (Product-building + rotbonds
    # for the whole 137k pool is wasteful; a few-thousand candidate draw fills every cell.)
    rows = []
    for cls in CLASSES:
        sub = pool[pool["cho_class"] == cls]
        take = min(args.cand_per_class, len(sub))
        cand = sub.iloc[rng.choice(len(sub), size=take, replace=False)]
        kept = 0
        for idx, smi in zip(cand["index"], cand["SMILES"]):
            if classify(smi) != "keep":
                continue
            bz = Th._make_benzoin_smiles(smi)
            if not bz:
                continue
            m = Chem.MolFromSmiles(bz)
            if m is None:
                continue
            rb = Th._mol_rotbonds(bz)
            rows.append({"index": int(idx), "SMILES": smi, "cho_class": cls,
                         "benzoin": bz, "product_rotbonds": int(rb),
                         "flex_bin": flex_bin(rb), "fp": gen.GetFingerprint(m)})
            kept += 1
        print(f"  {cls:16s}: {kept} valid candidates [{time.time()-t0:.0f}s]")

    cand = pd.DataFrame(rows)
    print("\nCandidate cell counts (class x flex_bin):")
    print(cand.groupby(["cho_class", "flex_bin"]).size().to_string())

    # MaxMin pick `per_cell` within each (class x flex bin) cell for spread.
    picks = []
    for cls in CLASSES:
        for lo, hi, bin_name in FLEX_BINS:
            cell = cand[(cand["cho_class"] == cls) & (cand["flex_bin"] == bin_name)]
            if len(cell) == 0:
                print(f"  WARN empty cell {cls}/{bin_name}")
                continue
            fps = list(cell["fp"])
            n = min(args.per_cell, len(fps))
            if len(fps) <= n:
                sel = list(range(len(fps)))
            else:
                picker = rdSimDivPickers.MaxMinPicker()
                sel = list(picker.LazyBitVectorPick(fps, len(fps), n, seed=args.seed))
            chosen = cell.iloc[sel].copy()
            chosen["sel_method"] = "maxmin"
            picks.append(chosen)

    out = pd.concat(picks, ignore_index=True)
    out = out.drop(columns=["fp"]).sort_values(["cho_class", "product_rotbonds"])
    cols = ["index", "SMILES", "cho_class", "benzoin", "product_rotbonds",
            "flex_bin", "sel_method"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(args.out, index=False)
    print(f"\nSelected {len(out)} molecules -> {args.out}")
    print(out.groupby(["cho_class", "flex_bin"]).size().to_string())
    print(f"product_rotbonds: min={out.product_rotbonds.min()} "
          f"med={int(out.product_rotbonds.median())} max={out.product_rotbonds.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
