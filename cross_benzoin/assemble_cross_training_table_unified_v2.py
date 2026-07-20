#!/usr/bin/env python3
"""
Track B (homo+cross unification), re-test at the round1-5 (17270-row) scale.
The first attempt (assemble_cross_training_table_unified.py, see
[[cross-round3-and-ensemble72-packaging-20260714]]) unified rounds1-3
(4120 rows, no mordred) with a 30,000-row stratified homo subsample and found
a real but modest gain on cross-row CV predictions (2.910 vs 2.966, +1.9%
relative). That test hasn't been re-run since round4/5 nearly quadrupled
cross's own data (4120->17270) and mordred descriptors were folded in --
both could plausibly change whether homo rows still help (more cross data of
its own may need homo's assist less) or hurt (mordred's higher dimensionality
may make homo's 30k rows dominate a larger feature space differently).

Generalizes assemble_cross_training_table_v3.py's --rounds mechanism (mordred
join included) to ALSO append the existing homo_unify_v1 sample (unchanged --
same 30k-row stratified subsample, same product BDE, so this isolates the
effect of cross's own data growth + mordred, not a homo-side change).

Usage
  python cross_benzoin/assemble_cross_training_table_unified_v2.py --rounds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
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
from assemble_cross_training_table_v3 import (  # noqa: E402
    ALD_MORDRED_CSV, DEFAULT_PRODUCT_MORDRED_CSVS, round_paths,
)

HOMO_PRODUCTS = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_products.csv"
HOMO_DFT = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_dft.csv"
HOMO_BDE = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_bde.csv"
HOMO_PRODUCT_MORDRED = REPO / "data/cross_benzoin/homo_v6/products_mordred_descriptors.csv"

OUT_DIR = REPO / "data/cross_benzoin/homo_unify"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", required=True)
    ap.add_argument("--out-tag", default=None,
                     help="output filename tag, default 'unified_v2_<N>rounds_mordred'")
    ap.add_argument("--product-mordred-csv", type=Path, nargs="+",
                     default=DEFAULT_PRODUCT_MORDRED_CSVS,
                     help="cross-side product mordred CSVs (see assemble_cross_training_table_v3.py)")
    args = ap.parse_args()
    tag = args.out_tag or f"unified_v2_{len(args.rounds)}rounds_mordred"
    out_parquet = OUT_DIR / f"cross_train_table_{tag}.parquet"
    out_csv = OUT_DIR / f"cross_train_table_{tag}.csv"

    ald = pd.read_csv(ALDEHYDES_CSV, low_memory=False)
    bde = pd.read_csv(ALD_BDE_CSV, usecols=["id", "bde_gxtb_kcal"])
    bde.loc[bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
    ald["id"] = pd.to_numeric(ald["id"], errors="coerce")
    bde["id"] = pd.to_numeric(bde["id"], errors="coerce")
    ald = ald.merge(bde, on="id", how="left")

    ald_mordred = pd.read_csv(ALD_MORDRED_CSV, low_memory=False)
    ald_mordred["id"] = pd.to_numeric(ald_mordred["id"], errors="coerce")
    mordred_cols = [c for c in ald_mordred.columns if c != "id"]
    ald = ald.merge(ald_mordred, on="id", how="left")
    print(f"aldehyde mordred join: {ald[mordred_cols[0]].notna().sum()}/{len(ald)} aldehydes "
          f"({len(mordred_cols)} cols)")

    ald["_canon"] = ald["smiles"].map(canon)
    ald_lookup = ald.drop_duplicates("_canon").set_index("_canon")[ALDEHYDE_FEATS + mordred_cols]

    rounds = []
    for n in args.rounds:
        products_csv, dft_csv, bde_path = round_paths(n)
        r = load_round(products_csv, dft_csv, bde_path, ald_lookup, f"round{n}")
        rounds.append(r)
    rh = load_round(HOMO_PRODUCTS, HOMO_DFT, HOMO_BDE, ald_lookup, "homo_unify_v1")
    rounds.append(rh)

    common_cols = set(rounds[0].columns)
    for r in rounds[1:]:
        common_cols &= set(r.columns)
    common_cols = [c for c in rounds[0].columns if c in common_cols]
    df = pd.concat([r[common_cols] for r in rounds], ignore_index=True)
    print(f"combined: {len(df)} rows ({df['round'].value_counts().to_dict()})")

    # product-side mordred: cross rounds' existing CSVs + homo's own (join on id, add product_ prefix)
    prod_mordred_frames = [pd.read_csv(p, low_memory=False) for p in args.product_mordred_csv if p.exists()]
    homo_pm = pd.read_csv(HOMO_PRODUCT_MORDRED, low_memory=False)
    prod_mordred_frames.append(homo_pm)
    prod_mordred = pd.concat(prod_mordred_frames, ignore_index=True).drop_duplicates("id")
    prod_mordred = prod_mordred.rename(
        columns={c: f"product_{c}" for c in prod_mordred.columns if c != "id"})
    df["id"] = df["id"].astype(str)
    prod_mordred["id"] = prod_mordred["id"].astype(str)
    df = df.merge(prod_mordred, on="id", how="left")
    prod_mordred_cols = [c for c in prod_mordred.columns if c != "id"]
    cov = df[prod_mordred_cols[0]].notna().sum() if prod_mordred_cols else 0
    print(f"product mordred join: {cov}/{len(df)} rows ({len(prod_mordred_cols)} cols)")

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

    keep_meta = ["id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
                 "donor_smiles", "acceptor_smiles", "smiles",
                 "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"]
    donor_mordred = [f"donor_{c}" for c in mordred_cols]
    acceptor_mordred = [f"acceptor_{c}" for c in mordred_cols]
    feat_cols = ([f"donor_{c}" for c in ALDEHYDE_FEATS] +
                 [f"acceptor_{c}" for c in ALDEHYDE_FEATS] +
                 [c for c in PRODUCT_FEATS if c in df.columns] +
                 list(donor2d.columns) + list(acc2d.columns) + list(prod2d.columns) +
                 [c for c in df.columns if c.startswith("interaction_")] +
                 donor_mordred + acceptor_mordred + prod_mordred_cols)
    feat_cols = list(dict.fromkeys(c for c in feat_cols if c in df.columns))
    out = df[keep_meta + feat_cols].copy()

    out.to_parquet(out_parquet, index=False)
    out.to_csv(out_csv, index=False)
    print(f"wrote {len(out)} rows x {len(feat_cols)} features -> {out_parquet}")
    print(f"  unordered pairs: {out['pair_key'].nunique()}")
    print(out["round"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
