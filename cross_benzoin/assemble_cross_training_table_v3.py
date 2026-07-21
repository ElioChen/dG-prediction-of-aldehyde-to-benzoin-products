#!/usr/bin/env python3
"""
Production training-table assembly, v3: generalizes assemble_cross_training_
table_combined_v2.py (which only knew about rounds 1-3, hardcoded) to any list
of rounds, AND folds in the mordred descriptor block that quick_cv_compare.py
validated as a real (not noise) gain -- MAE 2.965->2.921, R2 0.750->0.756 on
the 4120-row rounds1-3 table, see [[cross-round3-and-ensemble72-packaging-
20260714]]. Prior to this script mordred lived only in a one-off test table
(cross_train_table_3rounds_mordred_full.parquet); this makes it the default
so every future round/retrain gets it without re-deriving the join by hand.

Two mordred pieces, both reused from already-computed data (zero new compute
run by this script itself):
  - aldehyde-side (102 cols, donor_mordred_*/acceptor_mordred_*): FREE, sourced
    from the full 220k-aldehyde library file homo_v6/aldehydes_mordred_slim102.csv,
    joined the same way ALD_BDE_CSV already is (by aldehyde library `id`).
  - product-side (219 cols, product_mordred_*): genuinely-new-but-cheap compute
    already run for rounds 1-3 via add_mordred_cross_products.py (job 24642802),
    concatenated at cross_round3/rounds123_products_mordred.csv, joined by the
    products table's own `id` (a donor__acceptor InChIKey pair string, NOT a
    row index). Rounds without a matching mordred file simply get NaN for these
    columns (median-imputed downstream by train_cross_delta.py, same handling
    as any other optional block) -- this script does not require every round to
    have product mordred computed to run.

Legacy rounds 1-2 keep their historical hardcoded paths (inherited from
assemble_cross_training_table_combined.py's load_round()); round 3+ follow the
uniform cross_round{N}/cross_round{N}_dft_products.csv naming convention.

Usage
  python cross_benzoin/assemble_cross_training_table_v3.py --rounds 1 2 3
  python cross_benzoin/assemble_cross_training_table_v3.py --rounds 1 2 3 4 \
      --out-tag 4rounds
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
    ALDEHYDES_CSV, ALD_BDE_CSV,
    ROUND1_PRODUCTS, ROUND1_DFT, ROUND1_PRODUCT_BDE,
    ROUND2_PRODUCTS, ROUND2_DFT, ROUND2_PRODUCT_BDE_DIR,
    canon, load_round,
)

ALD_MORDRED_CSV = REPO / "data/cross_benzoin/homo_v6/aldehydes_mordred_slim102.csv"
DEFAULT_PRODUCT_MORDRED_CSVS = [
    REPO / "data/cross_benzoin/cross_round3/rounds123_products_mordred.csv",
]

LEGACY_PATHS = {
    1: (ROUND1_PRODUCTS, ROUND1_DFT, ROUND1_PRODUCT_BDE),
    2: (ROUND2_PRODUCTS, ROUND2_DFT, ROUND2_PRODUCT_BDE_DIR),
}


def round_paths(n: int):
    if n in LEGACY_PATHS:
        return LEGACY_PATHS[n]
    rtag = f"cross_round{n}"
    rdir = REPO / "data/cross_benzoin" / rtag
    return (rdir / f"{rtag}_dft_products.csv",
            REPO / f"data/raw/dft_sp_cross/{rtag}/{rtag}_dft_sp.csv",
            rdir / "bde_gxtb")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", required=True)
    ap.add_argument("--out-tag", default=None,
                     help="output filename tag, default '<N>rounds_mordred' for N=len(rounds)")
    ap.add_argument("--product-mordred-csv", type=Path, nargs="+",
                     default=DEFAULT_PRODUCT_MORDRED_CSVS,
                     help="one or more concatenated add_mordred_cross_products.py outputs, "
                          "joined on the products table's `id` column")
    args = ap.parse_args()
    tag = args.out_tag or f"{len(args.rounds)}rounds_mordred"
    out_dir = REPO / "data/cross_benzoin" / f"cross_round{max(args.rounds)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"cross_train_table_{tag}.parquet"
    out_csv = out_dir / f"cross_train_table_{tag}.csv"

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

    common_cols = set(rounds[0].columns)
    for r in rounds[1:]:
        common_cols &= set(r.columns)
    common_cols = [c for c in rounds[0].columns if c in common_cols]
    df = pd.concat([r[common_cols] for r in rounds], ignore_index=True)
    print(f"combined: {len(df)} rows ({df['round'].value_counts().to_dict()})")

    # product-side mordred (join on the products table's own `id`, may be NaN for
    # rounds not yet covered by a mordred products file -- left for median-fill)
    prod_mordred = pd.concat(
        [pd.read_csv(p, low_memory=False) for p in args.product_mordred_csv if p.exists()],
        ignore_index=True,
    ).drop_duplicates("id")
    prod_mordred = prod_mordred.rename(
        columns={c: f"product_{c}" for c in prod_mordred.columns if c != "id"})
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
