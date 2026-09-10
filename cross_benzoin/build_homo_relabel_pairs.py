#!/usr/bin/env python3
"""
Build the pair list for the HOMO full-library self-consistent relabel campaign
(rec_homo_relabel_worker.py --sample). One row per homo product with a valid
structure that does NOT already have a surviving 30k DFT label, plus a
front-loaded QC block of already-labelled pairs so early shards validate
against the surviving labels.

Columns (worker reads pid, prod_smiles, donor_smiles, and the optional
grp/new_scaffold_split/label/gxtb_stored):
  pid, donor_smiles, acceptor_smiles, prod_smiles,
  grp, new_scaffold_split, label, gxtb_stored, is_qc

Usage
  python cross_benzoin/build_homo_relabel_pairs.py \
      --qc-n 2000 \
      --out data/cross_benzoin/homo_standalone/homo_relabel_pairs.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
H = REPO / "data/cross_benzoin/homo_v6"
U = REPO / "data/cross_benzoin/homo_unify"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-n", type=int, default=2000,
                    help="front-loaded already-labelled pairs for a self-consistency check")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    prod = pd.read_csv(H / "products_all.csv", low_memory=False,
                       usecols=["id", "donor_smiles", "acceptor_smiles", "smiles",
                                "reaction_type", "error"])
    prod["id"] = prod["id"].astype(str)
    ok = (prod["error"].astype("string").fillna("") == "") & prod["smiles"].notna() \
        & prod["donor_smiles"].notna()
    prod = prod[ok].copy()
    print(f"homo products with valid structure: {len(prod)}")

    # surviving 30k labels
    lab = pd.read_csv(U / "homo_unify_v1_dft.csv")
    lab["id"] = lab["id"].astype(str)
    lab_map = dict(zip(lab["id"], lab["dG_orca_kcal"]))

    # g-xTB dG (stored) for repro diagnostics, from homo_unify_v1 (30k) where available
    try:
        hb = pd.read_csv(U / "homo_unify_v1_products.csv", low_memory=False,
                         usecols=["id", "dG_gxtb_kcal"])
        hb["id"] = hb["id"].astype(str)
        gx_map = dict(zip(hb["id"], hb["dG_gxtb_kcal"]))
    except Exception:
        gx_map = {}

    # scaffold-disjoint split (full-library)
    sp = pd.read_csv(H / "products_scaffold_split.csv")
    sp["id"] = sp["id"].astype(str)
    split_map = dict(zip(sp["id"], sp["scaffold_split"]))

    prod["prod_smiles"] = prod["smiles"]
    prod["grp"] = prod["reaction_type"]
    prod["new_scaffold_split"] = prod["id"].map(split_map)
    prod["label"] = prod["id"].map(lab_map)
    prod["gxtb_stored"] = prod["id"].map(gx_map)
    prod["is_qc"] = 0

    have_label = prod["label"].notna()
    todo = prod[~have_label].copy()
    labelled = prod[have_label].copy()
    print(f"  already-labelled: {len(labelled)}   to relabel: {len(todo)}")

    rng = np.random.default_rng(args.seed)
    qc_n = min(args.qc_n, len(labelled))
    qc_idx = rng.choice(labelled.index.to_numpy(), size=qc_n, replace=False)
    qc = labelled.loc[qc_idx].copy()
    qc["is_qc"] = 1

    # shuffle the to-relabel block so array segments are split-balanced
    todo = todo.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    cols = ["pid", "donor_smiles", "acceptor_smiles", "prod_smiles",
            "grp", "new_scaffold_split", "label", "gxtb_stored", "is_qc"]
    qc = qc.rename(columns={"id": "pid"})[cols]
    todo = todo.rename(columns={"id": "pid"})[cols]
    out = pd.concat([qc, todo], ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {len(out)} rows ({qc_n} QC + {len(todo)} relabel) -> {args.out}")
    print("split of relabel block:", todo["new_scaffold_split"].value_counts(dropna=False).to_dict())
    print("grp of relabel block:", todo["grp"].value_counts(dropna=False).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
