#!/usr/bin/env python3
"""
Round-3 active-learning step — combine round 1 (598 rows) + round 2 (1756
rows) + round 3's newly DFT-labeled subset (1766 rows, chosen by
bootstrap-ensemble uncertainty scored against the round1+2 combined model,
see [[cross-round3-and-ensemble72-packaging-20260714]]) into ONE training
table with the same 150-feature schema as the prior two combined tables.

Reuses assemble_cross_training_table_combined.py's load_round() (imported,
not re-implemented) for rounds 1 and 2; adds round 3 with the same function.

Does NOT touch round 2's existing cross_train_table_combined.{parquet,csv}
(preserve output history) -- writes to a new location under cross_round3/.

Usage
  python cross_benzoin/assemble_cross_training_table_combined_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

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
import numpy as np

ROUND3_PRODUCTS = REPO / "data/cross_benzoin/cross_round3/cross_round3_dft_products.csv"
ROUND3_DFT = REPO / "data/raw/dft_sp_cross/cross_round3/cross_round3_dft_sp.csv"
ROUND3_PRODUCT_BDE_DIR = REPO / "data/cross_benzoin/cross_round3/bde_gxtb"

OUT_PARQUET = REPO / "data/cross_benzoin/cross_round3/cross_train_table_3rounds.parquet"
OUT_CSV = REPO / "data/cross_benzoin/cross_round3/cross_train_table_3rounds.csv"


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

    common_cols = [c for c in r1.columns if c in r2.columns and c in r3.columns]
    df = pd.concat([r1[common_cols], r2[common_cols], r3[common_cols]], ignore_index=True)
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
