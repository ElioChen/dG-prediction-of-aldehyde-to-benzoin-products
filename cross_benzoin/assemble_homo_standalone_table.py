#!/usr/bin/env python3
"""
Standalone HOMO (self-condensation) dG training table -- homo rows ONLY, no
cross rounds, matched to the cross champion's 260-feature schema so
train_scaffold_disjoint.py / train_cross_gnn_arch_sweep.py consume it
unchanged and the holdout numbers are directly comparable to the cross
champion (blend MAE 2.215).

Schema match: aldehyde QM feats (donor/acceptor) + product QM feats + rdkit
2D blocks + interaction_* + the SAME 53 `product_mordred_*` columns the cross
champion uses (feature_list.json). No aldehyde mordred (champion has none).

Scope: the 30,000-row surviving-label homo subsample (homo_unify_v1). The
full ~219k homo library lost its DFT labels (physically gone; recompute is
the approved post-Tier-B campaign), so 30k is all that is trainable now.
Adds `new_scaffold_split` (train/test/validation) + `is_homo=1`.

Usage
  python cross_benzoin/assemble_homo_standalone_table.py \
      --out data/cross_benzoin/homo_standalone/homo_standalone_train_table_slim260.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "pipeline" / "compute"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble_cross_training_table import (  # noqa: E402
    ALDEHYDE_FEATS, PRODUCT_FEATS, MISMATCH_PAIRS, _rdkit_block,
)
from assemble_cross_training_table_combined import (  # noqa: E402
    ALDEHYDES_CSV, ALD_BDE_CSV, canon, load_round,
)

HOMO_PRODUCTS = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_products.csv"
HOMO_DFT = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_dft.csv"
HOMO_BDE = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_bde.csv"
HOMO_SPLIT = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_scaffold_split_lookup.csv"
HOMO_PRODUCT_MORDRED = REPO / "data/cross_benzoin/homo_v6/products_mordred_descriptors.csv"
CHAMPION_FEATURE_LIST = REPO / "data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    champ = json.loads(CHAMPION_FEATURE_LIST.read_text())
    pm_keep = [c[len("product_"):] for c in champ if c.startswith("product_mordred_")]
    print(f"champion product_mordred_ columns to keep: {len(pm_keep)}")

    # --- aldehyde-side lookup (QM feats only, no mordred), keyed by canonical SMILES ---
    ald = pd.read_csv(ALDEHYDES_CSV, low_memory=False)
    bde = pd.read_csv(ALD_BDE_CSV, usecols=["id", "bde_gxtb_kcal"])
    bde.loc[bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
    ald["id"] = pd.to_numeric(ald["id"], errors="coerce")
    bde["id"] = pd.to_numeric(bde["id"], errors="coerce")
    ald = ald.merge(bde, on="id", how="left")
    ald["_canon"] = ald["smiles"].map(canon)
    ald_lookup = ald.drop_duplicates("_canon").set_index("_canon")[ALDEHYDE_FEATS]

    # --- homo rows (donor==acceptor) via the shared load_round() ---
    df = load_round(HOMO_PRODUCTS, HOMO_DFT, HOMO_BDE, ald_lookup, "homo_unify_v1")
    print(f"homo rows with label + descriptors: {len(df)}")

    # --- product-side mordred: ONLY the 53 champion columns, product_ prefix ---
    pm = pd.read_csv(HOMO_PRODUCT_MORDRED, usecols=["id"] + pm_keep, low_memory=False)
    pm = pm.drop_duplicates("id")
    pm = pm.rename(columns={c: f"product_{c}" for c in pm_keep})
    df["id"] = df["id"].astype(str)
    pm["id"] = pm["id"].astype(str)
    df = df.merge(pm, on="id", how="left")
    pm_cols = [f"product_{c}" for c in pm_keep]
    print(f"product mordred join: {df[pm_cols[0]].notna().sum()}/{len(df)} ({len(pm_cols)} cols)")

    # --- rdkit 2D blocks + interaction terms (identical to unified_v2 / champion) ---
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

    df["donor_id"] = df["donor_id"].astype(str)
    df["acceptor_id"] = df["acceptor_id"].astype(str)
    df["pair_key"] = df.apply(lambda r: "__".join(sorted([r.donor_id, r.acceptor_id])), axis=1)
    df["is_homo"] = 1

    # --- scaffold-disjoint split from the homo_unify_v1 lookup ---
    sp = pd.read_csv(HOMO_SPLIT)
    sp["id"] = sp["id"].astype(str)
    df = df.merge(sp[["id", "scaffold_split"]], on="id", how="left")
    df = df.rename(columns={"scaffold_split": "new_scaffold_split"})
    n_missing = int(df["new_scaffold_split"].isna().sum())
    if n_missing:
        print(f"WARN: {n_missing} rows without a split label -> dropping")
        df = df[df["new_scaffold_split"].notna()].reset_index(drop=True)
    print("split:", df["new_scaffold_split"].value_counts().to_dict())

    keep_meta = ["id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
                 "is_homo", "new_scaffold_split",
                 "donor_smiles", "acceptor_smiles", "smiles",
                 "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"]
    feat_cols = ([f"donor_{c}" for c in ALDEHYDE_FEATS] +
                 [f"acceptor_{c}" for c in ALDEHYDE_FEATS] +
                 [c for c in PRODUCT_FEATS if c in df.columns] +
                 list(donor2d.columns) + list(acc2d.columns) + list(prod2d.columns) +
                 [c for c in df.columns if c.startswith("interaction_")] +
                 pm_cols)
    feat_cols = list(dict.fromkeys(c for c in feat_cols if c in df.columns))
    out = df[keep_meta + feat_cols].copy()

    out.to_parquet(args.out, index=False)
    out.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"\nwrote {len(out)} rows x {len(feat_cols)} features -> {args.out}")
    print(out["reaction_type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
