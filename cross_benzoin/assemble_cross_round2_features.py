#!/usr/bin/env python3
"""
Round-2 active-learning step 1 — assemble the SAME feature schema as
assemble_cross_training_table.py (donor_/acceptor_/product_/interaction_ blocks)
for the 4200-row round-2 batch, WITHOUT joining a DFT label (round 2 has none
yet — that's the point: score every row with the round-1-trained Δ-model +
an uncertainty estimate, then pick which subset actually gets DFT-SP'd).

Reuses round-1's aldehyde library (homo_v6/aldehydes_all.csv) and BDE cache —
round 2's aldehydes are all drawn from the same clean_v6 library, no new
aldehyde-side compute needed. Product BDE is round-2-specific (job 24620376).

Usage
  python cross_benzoin/assemble_cross_round2_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "pipeline" / "compute"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble_cross_training_table import (  # noqa: E402
    ALDEHYDE_FEATS, PRODUCT_FEATS, RDKIT_FEATS, MISMATCH_PAIRS, _rdkit_block,
)

PRODUCTS_CSV = REPO / "data/cross_benzoin/cross_round2/cross_round2_products.csv"
ALDEHYDES_CSV = REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv"
ALD_BDE_CSV = REPO / "data/cross_benzoin/homo_v6/aldehydes_bdfe_gxtb_descriptors.csv"
PRODUCT_BDE_DIR = REPO / "data/cross_benzoin/cross_round2/bde_gxtb"
OUT_PARQUET = REPO / "data/cross_benzoin/cross_round2/cross_round2_features.parquet"
OUT_CSV = REPO / "data/cross_benzoin/cross_round2/cross_round2_features.csv"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def main() -> int:
    prod = pd.read_csv(PRODUCTS_CSV, low_memory=False)
    ald = pd.read_csv(ALDEHYDES_CSV, low_memory=False)

    bde = pd.read_csv(ALD_BDE_CSV, usecols=["id", "bde_gxtb_kcal"])
    bde.loc[bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
    ald["id"] = pd.to_numeric(ald["id"], errors="coerce")
    bde["id"] = pd.to_numeric(bde["id"], errors="coerce")
    ald = ald.merge(bde, on="id", how="left")
    print(f"  donor/acceptor BDE join: {ald['bde_gxtb_kcal'].notna().sum()}/{len(ald)} aldehydes")

    df = prod[prod["error"].astype("string").fillna("") == ""].copy().reset_index(drop=True)
    print(f"round-2 products (error-free): {len(df)}/{len(prod)}")

    if PRODUCT_BDE_DIR.exists():
        pbde_files = sorted(PRODUCT_BDE_DIR.glob("chunk_*.csv"))
        if pbde_files:
            pbde = pd.concat([pd.read_csv(f) for f in pbde_files], ignore_index=True)
            pbde = pbde[["id", "bde_gxtb_kcal"]].drop_duplicates("id")
            pbde.loc[pbde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
            df = df.merge(pbde, on="id", how="left")
            print(f"  product BDE join: {df['bde_gxtb_kcal'].notna().sum()}/{len(df)} products "
                  f"({len(pbde_files)} chunk files)")
        else:
            df["bde_gxtb_kcal"] = np.nan
            print("  product BDE dir exists but no chunk files yet -- left as NaN")
    else:
        df["bde_gxtb_kcal"] = np.nan
        print("  product BDE not computed yet -- left as NaN (model will median-fill)")

    ald = ald.copy()
    ald["_canon"] = ald["smiles"].map(canon)
    ald_lookup = ald.drop_duplicates("_canon").set_index("_canon")[ALDEHYDE_FEATS]

    df["_donor_canon"] = df["donor_smiles"].map(canon)
    df["_acceptor_canon"] = df["acceptor_smiles"].map(canon)
    ald_d = ald_lookup.add_prefix("donor_").reset_index().rename(columns={"_canon": "_donor_canon"})
    ald_a = ald_lookup.add_prefix("acceptor_").reset_index().rename(columns={"_canon": "_acceptor_canon"})
    df = df.merge(ald_d, on="_donor_canon", how="left").merge(ald_a, on="_acceptor_canon", how="left")
    miss_d = df["donor_G_gxtb"].isna().sum()
    miss_a = df["acceptor_G_gxtb"].isna().sum()
    if miss_d or miss_a:
        print(f"WARN: {miss_d} rows missing donor aldehyde descriptors, "
              f"{miss_a} missing acceptor -- dropping")
        df = df[df["donor_G_gxtb"].notna() & df["acceptor_G_gxtb"].notna()].copy()

    donor2d = _rdkit_block(df["donor_smiles"], "donor")
    acc2d = _rdkit_block(df["acceptor_smiles"], "acceptor")
    prod2d = _rdkit_block(df["smiles"], "product")
    df = pd.concat([df.reset_index(drop=True), donor2d, acc2d, prod2d], axis=1)

    df["interaction_gap_HOMOd_LUMOa"] = df["donor_xtb_HOMO"] - df["acceptor_xtb_LUMO"]
    df["interaction_fukui_match"] = df["donor_fukui_minus_CHO_C"] * df["acceptor_fukui_plus_CHO_C"]
    for feat in MISMATCH_PAIRS:
        d, a = f"donor_{feat}", f"acceptor_{feat}"
        if d in df.columns and a in df.columns:
            df[f"interaction_absdiff_{feat}"] = (df[d] - df[a]).abs()

    df["pair_key"] = df.apply(lambda r: "__".join(sorted([r.donor_id, r.acceptor_id])), axis=1)

    keep_meta = ["id", "donor_id", "acceptor_id", "pair_key", "reaction_type",
                 "donor_smiles", "acceptor_smiles", "smiles",
                 "dG_xtb_kcal", "dG_gxtb_kcal"]
    feat_cols = ([f"donor_{c}" for c in ALDEHYDE_FEATS] +
                 [f"acceptor_{c}" for c in ALDEHYDE_FEATS] +
                 [c for c in PRODUCT_FEATS if c in df.columns] +
                 list(donor2d.columns) + list(acc2d.columns) + list(prod2d.columns) +
                 [c for c in df.columns if c.startswith("interaction_")])
    feat_cols = list(dict.fromkeys(feat_cols))
    out = df[[c for c in keep_meta if c in df.columns] + feat_cols].copy()

    out.to_parquet(OUT_PARQUET, index=False)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out)} rows x {len(feat_cols)} features -> {OUT_PARQUET}")
    print(f"  unordered pairs: {out['pair_key'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
