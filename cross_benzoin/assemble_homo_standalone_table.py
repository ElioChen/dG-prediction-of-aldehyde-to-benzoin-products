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
sys.path.insert(0, str(REPO / "pipeline" / "bde"))
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

# --full-library mode (HANDOFF_20260910 Sec1.3, post homo-SP-relabel-drain): the
# full ~184k product library + its own aldehyde/split/bde/mordred caches, with
# targets/baseline coming from the SP relabel campaign's merged labels
# (merge_homo_sp.py output) instead of the 30k homo_unify_v1 subsample.
FULL_ALDEHYDES = REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv"
FULL_PRODUCTS = REPO / "data/cross_benzoin/homo_v6/products_all.csv"
FULL_PRODUCT_BDE = REPO / "data/cross_benzoin/homo_v6/products_bdfe_gxtb_descriptors.csv"
FULL_SPLIT = REPO / "data/cross_benzoin/homo_v6/products_scaffold_split.csv"


def build_full_library(labels_path: Path, pm_keep: list[str]) -> pd.DataFrame:
    """Full ~184k homo library, targets/baseline from the SP relabel labels."""
    from qc import norm_id  # pipeline/bde/qc.py -- "2.0"-style recovered-CSV id bug

    prod = pd.read_csv(FULL_PRODUCTS, low_memory=False)
    prod["id"] = norm_id(prod["id"])
    err = prod["error"].astype(str).str.strip()
    prod = prod[err.eq("") | err.eq("nan")].copy()
    print(f"products_all.csv: {len(prod)} rows with no worker error")

    bde = pd.read_csv(FULL_PRODUCT_BDE, usecols=["id", "bde_gxtb_kcal"])
    bde["id"] = norm_id(bde["id"])
    bde.loc[bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
    prod = prod.merge(bde, on="id", how="left")

    labels = pd.read_csv(labels_path, dtype={"id": str})
    labels["id"] = norm_id(labels["id"])
    df = prod.merge(labels, on="id", how="inner")
    print(f"products with usable SP relabel: {len(df)}/{len(prod)}")

    # --- aldehyde-side features: homo => donor==acceptor==the product's own aldehyde id ---
    ald = pd.read_csv(FULL_ALDEHYDES, low_memory=False)
    ald["id"] = norm_id(ald["id"])
    ald_bde = pd.read_csv(ALD_BDE_CSV, usecols=["id", "bde_gxtb_kcal"])
    ald_bde["id"] = norm_id(ald_bde["id"])
    ald_bde.loc[ald_bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
    ald = ald.merge(ald_bde, on="id", how="left")
    ald_lookup = ald.drop_duplicates("id").set_index("id")[ALDEHYDE_FEATS]
    df = df.reset_index(drop=True)
    for prefix in ("donor", "acceptor"):
        block = ald_lookup.reindex(df["id"]).reset_index(drop=True)
        block.columns = [f"{prefix}_{c}" for c in block.columns]
        df = pd.concat([df, block], axis=1)
    n_ald_missing = int(df["donor_G_xtb"].isna().sum())
    if n_ald_missing:
        print(f"WARN: {n_ald_missing} rows with no aldehyde-feature match -> dropping")
        df = df[df["donor_G_xtb"].notna()].reset_index(drop=True)

    # --- product-side mordred: ONLY the champion columns, product_ prefix ---
    pm = pd.read_csv(HOMO_PRODUCT_MORDRED, usecols=["id"] + pm_keep, low_memory=False)
    pm["id"] = norm_id(pm["id"])
    pm = pm.drop_duplicates("id")
    pm = pm.rename(columns={c: f"product_{c}" for c in pm_keep})
    df = df.merge(pm, on="id", how="left")
    pm_cols = [f"product_{c}" for c in pm_keep]
    print(f"product mordred join: {df[pm_cols[0]].notna().sum()}/{len(df)} ({len(pm_cols)} cols)")

    # --- rdkit 2D blocks + interaction terms (identical formulas to the 30k path) ---
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

    df["donor_id"] = df["id"]
    df["acceptor_id"] = df["id"]
    df["pair_key"] = df["id"] + "__" + df["id"]
    df["is_homo"] = 1
    df["reaction_type"] = "homo_full_sp"
    df["round"] = "homo_full_sp"

    # --- full-library scaffold-disjoint split ---
    sp = pd.read_csv(FULL_SPLIT, usecols=["id", "scaffold_split"])
    sp["id"] = norm_id(sp["id"])
    sp = sp.drop_duplicates("id")
    df = df.merge(sp, on="id", how="left")
    df = df.rename(columns={"scaffold_split": "new_scaffold_split"})
    n_missing = int(df["new_scaffold_split"].isna().sum())
    if n_missing:
        print(f"WARN: {n_missing} rows without a split label -> dropping")
        df = df[df["new_scaffold_split"].notna()].reset_index(drop=True)
    print("split:", df["new_scaffold_split"].value_counts().to_dict())

    # --- targets: r2SCAN-3c relabel is the new dG_orca_kcal; B97-3c is the Delta baseline ---
    df = df.rename(columns={"dG_r2scan_kcal": "dG_orca_kcal"})
    df["dG_gxtb_kcal"] = np.nan  # not computed by the SP-only route

    keep_meta = ["id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
                 "is_homo", "new_scaffold_split",
                 "donor_smiles", "acceptor_smiles", "smiles",
                 "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal", "dG_b973c_kcal"]
    feat_cols = ([f"donor_{c}" for c in ALDEHYDE_FEATS] +
                 [f"acceptor_{c}" for c in ALDEHYDE_FEATS] +
                 [c for c in PRODUCT_FEATS if c in df.columns] +
                 list(donor2d.columns) + list(acc2d.columns) + list(prod2d.columns) +
                 [c for c in df.columns if c.startswith("interaction_")] +
                 pm_cols)
    feat_cols = list(dict.fromkeys(c for c in feat_cols if c in df.columns))
    return df[keep_meta + feat_cols].copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-library", action="store_true",
                     help="assemble the full ~184k library instead of the 30k homo_unify_v1 subsample")
    ap.add_argument("--labels", type=Path,
                     help="full-library mode only: merge_homo_sp.py's <prefix>_labels.csv "
                          "(id,dG_r2scan_kcal,dG_b973c_kcal)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    champ = json.loads(CHAMPION_FEATURE_LIST.read_text())
    pm_keep = [c[len("product_"):] for c in champ if c.startswith("product_mordred_")]
    print(f"champion product_mordred_ columns to keep: {len(pm_keep)}")

    if args.full_library:
        if not args.labels:
            ap.error("--full-library requires --labels")
        out = build_full_library(args.labels, pm_keep)
        out.to_parquet(args.out, index=False)
        out.to_csv(args.out.with_suffix(".csv"), index=False)
        print(f"\nwrote {len(out)} rows x {len(out.columns) - 15} features -> {args.out}")
        print(out["reaction_type"].value_counts().to_string())
        return 0

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
