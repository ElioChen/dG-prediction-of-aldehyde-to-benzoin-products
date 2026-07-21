#!/usr/bin/env python3
"""
Generalized version of assemble_cross_round2_features.py -- assembles the
SAME 150-feature schema as assemble_cross_training_table.py for any round's
unlabeled products (no DFT join), parameterized by --round instead of one
hardcoded round number. Used starting round 3 so a 4th+ round doesn't need
yet another near-duplicate file.

Reuses round-1's aldehyde library (homo_v6/aldehydes_all.csv) and BDE cache
for every round -- all rounds draw aldehydes from the same clean_v6 library.

Usage
  python cross_benzoin/assemble_cross_round_features.py --round 3
"""
from __future__ import annotations

import argparse
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
# aldehyde-side mordred: free, full-220k-library, same file assemble_cross_training_table_v3.py
# uses for the labeled tables -- always joined so a round's candidate-scoring schema matches
# whatever mordred-augmented model it's being scored against (see [[cross-round3-and-
# ensemble72-packaging-20260714]]).
ALD_MORDRED_CSV = REPO / "data/cross_benzoin/homo_v6/aldehydes_mordred_slim102.csv"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None,
                     help="round number for the standard cross_round{N} naming convention")
    ap.add_argument("--tag", default=None,
                     help="override rtag/rdir directly for non-round-numbered pools "
                          "(e.g. --tag screen10k -> data/cross_benzoin/screen10k/)")
    ap.add_argument("--products-csv", type=Path, default=None,
                     help="override default {rtag}/{rtag}_products.csv "
                          "(e.g. a *_final.csv after a disk-quota-retry merge)")
    ap.add_argument("--product-mordred-csv", type=Path, nargs="*", default=[],
                     help="add_mordred_cross_products.py concatenated output(s), joined on `id`")
    ap.add_argument("--allow-missing-mordred", action="store_true",
                     help="proceed even if product_mordred_* can't be found/computed -- the "
                          "resulting table will be missing ~53 of the champion's 260 features "
                          "and CANNOT be scored against it. Default is to auto-discover "
                          "{rdir}/mordred_products/chunk_*.csv, and hard-fail if that's also "
                          "absent, since silently omitting mordred (found 2026-07-20 on "
                          "round8) produces a table that LOOKS complete but silently fails "
                          "downstream scoring.")
    args = ap.parse_args()
    if args.round is None and args.tag is None:
        raise SystemExit("must pass --round N or --tag NAME")
    rtag = args.tag or f"cross_round{args.round}"
    rdir = REPO / "data/cross_benzoin" / rtag
    products_csv = args.products_csv or (rdir / f"{rtag}_products.csv")
    product_bde_dir = rdir / "bde_gxtb"
    out_tag = "features" if args.products_csv is None else f"features_{args.products_csv.stem}"
    out_parquet = rdir / f"{rtag}_{out_tag}.parquet"
    out_csv = rdir / f"{rtag}_{out_tag}.csv"

    prod = pd.read_csv(products_csv, low_memory=False)
    ald = pd.read_csv(ALDEHYDES_CSV, low_memory=False)

    bde = pd.read_csv(ALD_BDE_CSV, usecols=["id", "bde_gxtb_kcal"])
    bde.loc[bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
    ald["id"] = pd.to_numeric(ald["id"], errors="coerce")
    bde["id"] = pd.to_numeric(bde["id"], errors="coerce")
    ald = ald.merge(bde, on="id", how="left")
    print(f"  donor/acceptor BDE join: {ald['bde_gxtb_kcal'].notna().sum()}/{len(ald)} aldehydes")

    ald_mordred = pd.read_csv(ALD_MORDRED_CSV, low_memory=False)
    ald_mordred["id"] = pd.to_numeric(ald_mordred["id"], errors="coerce")
    mordred_cols = [c for c in ald_mordred.columns if c != "id"]
    ald = ald.merge(ald_mordred, on="id", how="left")
    print(f"  aldehyde mordred join: {ald[mordred_cols[0]].notna().sum()}/{len(ald)} aldehydes "
          f"({len(mordred_cols)} cols)")

    df = prod[prod["error"].astype("string").fillna("") == ""].copy().reset_index(drop=True)
    print(f"{rtag} products (error-free): {len(df)}/{len(prod)}")

    if product_bde_dir.exists():
        pbde_files = sorted(product_bde_dir.glob("chunk_*.csv"))
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

    mordred_csvs = args.product_mordred_csv
    if not mordred_csvs:
        auto = sorted((rdir / "mordred_products").glob("chunk_*.csv"))
        if auto:
            print(f"  --product-mordred-csv not given, auto-discovered {len(auto)} files "
                  f"under {rdir/'mordred_products'}")
            mordred_csvs = auto
        elif not args.allow_missing_mordred:
            raise SystemExit(
                f"ERROR: no --product-mordred-csv given and no {rdir/'mordred_products'}/"
                f"chunk_*.csv found. product_mordred_* is a REQUIRED part of the champion's "
                f"260-feature schema (confirmed real gain, folded into production since "
                f"round6/7) -- run add_mordred_cross_products.py first (see "
                f"cross_benzoin/slurm/submit_mordred_cross_products.sh), or pass "
                f"--allow-missing-mordred to proceed anyway with an incomplete table."
            )
    if mordred_csvs:
        prod_mordred = pd.concat(
            [pd.read_csv(p, low_memory=False) for p in mordred_csvs if p.exists()],
            ignore_index=True,
        ).drop_duplicates("id")
        prod_mordred = prod_mordred.rename(
            columns={c: f"product_{c}" for c in prod_mordred.columns if c != "id"})
        df = df.merge(prod_mordred, on="id", how="left")
        prod_mordred_cols = [c for c in prod_mordred.columns if c != "id"]
        cov = df[prod_mordred_cols[0]].notna().sum() if prod_mordred_cols else 0
        print(f"  product mordred join: {cov}/{len(df)} rows ({len(prod_mordred_cols)} cols)")
    else:
        prod_mordred_cols = []
        print("  WARNING: proceeding with product_mordred_* cols omitted "
              "(--allow-missing-mordred) -- table cannot be scored against the champion")

    ald = ald.copy()
    ald["_canon"] = ald["smiles"].map(canon)
    ald_lookup = ald.drop_duplicates("_canon").set_index("_canon")[ALDEHYDE_FEATS + mordred_cols]

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
    donor_mordred = [f"donor_{c}" for c in mordred_cols]
    acceptor_mordred = [f"acceptor_{c}" for c in mordred_cols]
    feat_cols = ([f"donor_{c}" for c in ALDEHYDE_FEATS] +
                 [f"acceptor_{c}" for c in ALDEHYDE_FEATS] +
                 [c for c in PRODUCT_FEATS if c in df.columns] +
                 list(donor2d.columns) + list(acc2d.columns) + list(prod2d.columns) +
                 [c for c in df.columns if c.startswith("interaction_")] +
                 donor_mordred + acceptor_mordred + prod_mordred_cols)
    feat_cols = list(dict.fromkeys(c for c in feat_cols if c in df.columns))
    out = df[[c for c in keep_meta if c in df.columns] + feat_cols].copy()

    out.to_parquet(out_parquet, index=False)
    out.to_csv(out_csv, index=False)
    print(f"wrote {len(out)} rows x {len(feat_cols)} features -> {out_parquet}")
    print(f"  unordered pairs: {out['pair_key'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
