#!/usr/bin/env python3
"""
Track B (homo+cross unification), first attempt — merge cross rounds 1-3
(4120 rows) with a 30,000-row stratified homo subsample
(`build_homo_for_unification.py`, equal aliph-aliph/carbo-carbo/hetero-hetero)
into ONE 150-feature training table, using the same `load_round()` reused
from `assemble_cross_training_table_combined.py` for every source.

Homo rows carry `round="homo_unify_v1"` so they're easy to isolate/exclude in
later analysis (e.g. to check the cross-only frozen holdout is unaffected).

Usage
  python cross_benzoin/assemble_cross_training_table_unified.py
"""
from __future__ import annotations

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
from assemble_cross_training_table_combined_v2 import (  # noqa: E402
    ROUND3_PRODUCTS, ROUND3_DFT, ROUND3_PRODUCT_BDE_DIR,
)

HOMO_PRODUCTS = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_products.csv"
HOMO_DFT = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_dft.csv"
HOMO_BDE = REPO / "data/cross_benzoin/homo_unify/homo_unify_v1_bde.csv"

OUT_PARQUET = REPO / "data/cross_benzoin/homo_unify/cross_train_table_unified_v1.parquet"
OUT_CSV = REPO / "data/cross_benzoin/homo_unify/cross_train_table_unified_v1.csv"


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
    r3 = load_round(ROUND3_PRODUCTS, ROUND3_DFT, ROUND3_PRODUCT_BDE_DIR, ald_lookup, "round3")
    rh = load_round(HOMO_PRODUCTS, HOMO_DFT, HOMO_BDE, ald_lookup, "homo_unify_v1")

    common_cols = [c for c in r1.columns if all(c in r.columns for r in (r2, r3, rh))]
    df = pd.concat([r1[common_cols], r2[common_cols], r3[common_cols], rh[common_cols]],
                   ignore_index=True)
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

    df["id"] = df["id"].astype(str)
    df["donor_id"] = df["donor_id"].astype(str)
    df["acceptor_id"] = df["acceptor_id"].astype(str)
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
    print("\nreaction_type distribution:")
    print(out["reaction_type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
