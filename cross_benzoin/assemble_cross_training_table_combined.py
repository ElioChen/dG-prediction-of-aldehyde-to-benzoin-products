#!/usr/bin/env python3
"""
Round-2 active-learning step 3 — combine round 1 (cross_pilot_v1, 598 rows/
299 pairs, uniform-diversity sampled) with round 2's newly DFT-labeled subset
(880 pairs / 1756 rows, chosen by bootstrap-ensemble uncertainty, see
[[cross-round2-active-learning]]) into ONE training table with the exact same
150-feature schema `assemble_cross_training_table.py` built for round 1 alone.

Does NOT touch round 1's existing `cross_train_table.{parquet,csv}` (preserve
output history) -- writes to a new location under cross_round2/.

Usage
  python cross_benzoin/assemble_cross_training_table_combined.py
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

ALDEHYDES_CSV = REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv"
ALD_BDE_CSV = REPO / "data/cross_benzoin/homo_v6/aldehydes_bdfe_gxtb_descriptors.csv"

ROUND1_PRODUCTS = REPO / "data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_products.csv"
ROUND1_DFT = REPO / "data/raw/dft_sp_cross/cross_pilot_v1_dft_sp.csv"
ROUND1_PRODUCT_BDE = REPO / "data/cross_benzoin/cross_pilot_v1/bde_gxtb/cross_pilot_v1_product_bde_gxtb.csv"

ROUND2_PRODUCTS = REPO / "data/cross_benzoin/cross_round2/cross_round2_dft_products.csv"
ROUND2_DFT = REPO / "data/raw/dft_sp_cross/cross_round2/cross_round2_dft_sp.csv"
ROUND2_PRODUCT_BDE_DIR = REPO / "data/cross_benzoin/cross_round2/bde_gxtb"

OUT_PARQUET = REPO / "data/cross_benzoin/cross_round2/cross_train_table_combined.parquet"
OUT_CSV = REPO / "data/cross_benzoin/cross_round2/cross_train_table_combined.csv"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def load_round(products_csv: Path, dft_csv: Path, product_bde_csv_or_dir: Path,
               ald_lookup: pd.DataFrame, round_tag: str) -> pd.DataFrame:
    prod = pd.read_csv(products_csv, low_memory=False)
    dft = pd.read_csv(dft_csv)
    prod = prod[prod["error"].astype("string").fillna("") == ""].copy()
    if "error" in dft.columns:
        n_dft_err = (dft["error"].astype("string").fillna("") != "").sum()
        if n_dft_err:
            print(f"  [{round_tag}] dropping {n_dft_err} DFT-SP row(s) with a non-empty "
                  f"error (e.g. orca_sp_failed) before the label join")
        dft = dft[dft["error"].astype("string").fillna("") == ""].copy()

    if product_bde_csv_or_dir.is_dir():
        files = sorted(product_bde_csv_or_dir.glob("chunk_*.csv"))
        pbde = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        pbde = pd.read_csv(product_bde_csv_or_dir)
    pbde = pbde[["id", "bde_gxtb_kcal"]].drop_duplicates("id")
    pbde.loc[pbde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
    prod = prod.merge(pbde, on="id", how="left")

    df = prod.merge(dft[["id", "dG_orca_kcal"]], on="id", how="inner")
    print(f"  [{round_tag}] products (error-free): {len(prod)} -> with DFT label: {len(df)}  "
          f"(product BDE coverage {df['bde_gxtb_kcal'].notna().sum()}/{len(df)})")

    df["_donor_canon"] = df["donor_smiles"].map(canon)
    df["_acceptor_canon"] = df["acceptor_smiles"].map(canon)
    ald_d = ald_lookup.add_prefix("donor_").reset_index().rename(columns={"_canon": "_donor_canon"})
    ald_a = ald_lookup.add_prefix("acceptor_").reset_index().rename(columns={"_canon": "_acceptor_canon"})
    df = df.merge(ald_d, on="_donor_canon", how="left").merge(ald_a, on="_acceptor_canon", how="left")
    miss_d = df["donor_G_gxtb"].isna().sum()
    miss_a = df["acceptor_G_gxtb"].isna().sum()
    if miss_d or miss_a:
        print(f"  [{round_tag}] WARN: {miss_d} rows missing donor descriptors, "
              f"{miss_a} missing acceptor -- dropping")
        df = df[df["donor_G_gxtb"].notna() & df["acceptor_G_gxtb"].notna()].copy()
    df["round"] = round_tag
    return df


def main() -> int:
    ald = pd.read_csv(ALDEHYDES_CSV, low_memory=False)
    bde = pd.read_csv(ALD_BDE_CSV, usecols=["id", "bde_gxtb_kcal"])
    bde.loc[bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
    ald["id"] = pd.to_numeric(ald["id"], errors="coerce")
    bde["id"] = pd.to_numeric(bde["id"], errors="coerce")
    ald = ald.merge(bde, on="id", how="left")
    ald["_canon"] = ald["smiles"].map(canon)
    ald_lookup = ald.drop_duplicates("_canon").set_index("_canon")[ALDEHYDE_FEATS]

    r1 = load_round(ROUND1_PRODUCTS, ROUND1_DFT, ROUND1_PRODUCT_BDE, ald_lookup, "round1")
    r2 = load_round(ROUND2_PRODUCTS, ROUND2_DFT, ROUND2_PRODUCT_BDE_DIR, ald_lookup, "round2")

    common_cols = [c for c in r1.columns if c in r2.columns]
    df = pd.concat([r1[common_cols], r2[common_cols]], ignore_index=True)
    print(f"combined: {len(df)} rows ({df['round'].value_counts().to_dict()})")

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

    keep_meta = ["id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
                 "donor_smiles", "acceptor_smiles", "smiles",
                 "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"]
    feat_cols = ([f"donor_{c}" for c in ALDEHYDE_FEATS] +
                 [f"acceptor_{c}" for c in ALDEHYDE_FEATS] +
                 [c for c in PRODUCT_FEATS if c in df.columns] +
                 list(donor2d.columns) + list(acc2d.columns) + list(prod2d.columns) +
                 [c for c in df.columns if c.startswith("interaction_")])
    feat_cols = list(dict.fromkeys(feat_cols))
    out = df[keep_meta + feat_cols].copy()

    out.to_parquet(OUT_PARQUET, index=False)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out)} rows x {len(feat_cols)} features -> {OUT_PARQUET}")
    print(f"  unordered pairs: {out['pair_key'].nunique()}")
    print(out["round"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
