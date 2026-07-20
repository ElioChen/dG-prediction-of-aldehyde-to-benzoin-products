#!/usr/bin/env python3
"""
Phase 3 step 1 — assemble the role-aware cross-benzoin training table.

Joins, per DESCRIPTOR_POLICY_CROSS.md:
  donor_*      aldehyde descriptors (homo_v6/aldehydes_all.csv) on donor_id
  acceptor_*   same table on acceptor_id
  product_*    already role-aware columns in cross_pilot_v1_products.csv
               (mulliken_ketC/carbC, wbo_*, fukui_*, vbur_*, sterimol_*, ...)
  interaction_* donor/acceptor complementarity terms the doc calls out:
               HOMO(D)-LUMO(A) gap, Fukui-minus(D)*Fukui-plus(A), and
               |donor_x - acceptor_x| mismatch for shared descriptors

2D (geometry-free) descriptors for donor/acceptor/product are computed in-process
via ald_descriptors.calc_rdkit (no compute campaign needed — SMILES only).

Target: dG_orca_kcal (r2SCAN-3c DFT-SP, job 24609263). Baseline: dG_gxtb_kcal
(already in the products table). ADCH/QTAIM columns are all-null for this pilot
(Multiwfn wasn't run) and are dropped, matching the policy's "optional/expensive,
don't back-fill without an ablation payoff" guidance.

Usage
  python cross_benzoin/assemble_cross_training_table.py
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
from ald_descriptors import calc_rdkit  # noqa: E402

PRODUCTS_CSV = REPO / "data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_products.csv"
DFT_CSV = REPO / "data/raw/dft_sp_cross/cross_pilot_v1_dft_sp.csv"
ALDEHYDES_CSV = REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv"
# g-xTB BDE (per-aldehyde, C-H alpha to the carbonyl): the SINGLE highest-SHAP
# feature in the full-library homo champion (MAE 1.525->1.503, see
# [[bde-descriptor-idea]] / [[gxtb-dft-correction-champion]] -- "SHAP: BDE >>
# BDFE, drop BDFE"). Already computed for the whole 220k-aldehyde library, so
# reusing it here for donor_/acceptor_ is zero new compute -- the one piece of
# homo-model insight that transfers to cross as a feature (not a full transfer-
# learned encoder) with no extra cost. BDFE deliberately excluded (expensive
# --ohess, and the homo ablation found it added nothing over BDE alone).
ALD_BDE_CSV = REPO / "data/cross_benzoin/homo_v6/aldehydes_bdfe_gxtb_descriptors.csv"
# Product-side g-xTB BDE (new ketC-carbC bond) -- genuinely new cross-specific compute,
# job 24616599, see calc_bde_gxtb_product_cross.py. This is the piece donor/acceptor BDE
# alone couldn't substitute for (see cross-pilot-dft-sp-validated memory's null result).
PRODUCT_BDE_CSV = REPO / "data/cross_benzoin/cross_pilot_v1/bde_gxtb/cross_pilot_v1_product_bde_gxtb.csv"
OUT_PARQUET = REPO / "data/cross_benzoin/cross_pilot_v1/cross_train_table.parquet"
OUT_CSV = REPO / "data/cross_benzoin/cross_pilot_v1/cross_train_table.csv"

# Product-side columns already computed by cb_featurize.py, already role-aware
# (ketC/ketO = donor-derived ketone site; carbC/hydO/hydH = acceptor-derived
# carbinol site). ADCH/QTAIM excluded: all-null in this pilot (no --multiwfn).
PRODUCT_FEATS = [
    "xtb_energy", "xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA",
    "xtb_mu", "xtb_eta", "xtb_omega", "xtb_dipole",
    "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO", "mulliken_hydH",
    "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb",
    "fukui_plus_ketC", "fukui_minus_ketC", "fukui_0_ketC", "dual_ketC",
    "fukui_plus_carbC", "fukui_minus_carbC", "fukui_0_carbC", "dual_carbC",
    "pa_ketO", "vbur_ketC", "vbur_carbC",
    "sterimol_L", "sterimol_B1", "sterimol_B5", "SASA_total", "P_int",
    "hb_dist", "hb_angle", "dih_core", "bde_gxtb_kcal",
]

# Per-aldehyde columns in aldehydes_all.csv to carry into donor_*/acceptor_*.
ALDEHYDE_FEATS = [
    "G_xtb", "G_gxtb", "xtb_energy", "xtb_HOMO", "xtb_LUMO", "xtb_gap",
    "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta", "xtb_omega", "xtb_dipole",
    "mulliken_CHO_C", "mulliken_CHO_O",
    "fukui_plus_CHO_C", "fukui_minus_CHO_C", "fukui_0_CHO_C", "dual_descriptor_CHO_C",
    "wbo_CO", "pa_CHO_O", "vbur_CHO_C",
    "sterimol_L", "sterimol_B1", "sterimol_B5", "SASA_total", "P_int",
    "bde_gxtb_kcal",
]

RDKIT_FEATS = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "ArRings",
               "ArHetRings", "AlRings", "Rings", "Heteroatoms", "FractionCSP3",
               "BertzCT", "Kappa2", "NumStereocenters", "n_CHO"]

# Shared descriptors used for the |donor - acceptor| mismatch interaction block
# (steric/electronic complementarity the doc says homo-only data can't teach).
MISMATCH_PAIRS = ["xtb_gap", "xtb_dipole", "sterimol_L", "sterimol_B1",
                   "sterimol_B5", "SASA_total", "MW", "TPSA"]


def _rdkit_block(smiles: pd.Series, prefix: str) -> pd.DataFrame:
    cache: dict[str, dict] = {}
    rows = []
    for s in smiles:
        if s not in cache:
            cache[s] = calc_rdkit(s)
        rows.append(cache[s])
    df = pd.DataFrame(rows)[RDKIT_FEATS]
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df.reset_index(drop=True)


def main() -> int:
    prod = pd.read_csv(PRODUCTS_CSV)
    dft = pd.read_csv(DFT_CSV)
    ald = pd.read_csv(ALDEHYDES_CSV, low_memory=False)

    # Fold in the g-xTB BDE (same numeric row-index "id" scheme as aldehydes_all.csv).
    bde = pd.read_csv(ALD_BDE_CSV, usecols=["id", "bde_gxtb_kcal"])
    bde.loc[bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan  # same outlier guard as the homo pipeline
    ald["id"] = pd.to_numeric(ald["id"], errors="coerce")
    bde["id"] = pd.to_numeric(bde["id"], errors="coerce")
    ald = ald.merge(bde, on="id", how="left")
    print(f"  BDE join: {ald['bde_gxtb_kcal'].notna().sum()}/{len(ald)} aldehydes have g-xTB BDE")

    prod = prod[prod["error"].astype("string").fillna("") == ""].copy()
    if PRODUCT_BDE_CSV.exists():
        pbde = pd.read_csv(PRODUCT_BDE_CSV, usecols=["id", "bde_gxtb_kcal"])
        pbde.loc[pbde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
        prod = prod.merge(pbde, on="id", how="left")
        print(f"  product BDE join: {prod['bde_gxtb_kcal'].notna().sum()}/{len(prod)} products have g-xTB BDE")
    df = prod.merge(dft[["id", "dG_orca_kcal"]], on="id", how="inner")
    print(f"products (error-free): {len(prod)}  ->  with DFT label: {len(df)}")

    # aldehydes_all.csv's "id" is a numeric row-index into aldehydes_clean_v6.csv,
    # NOT the InChIKey used for donor_id/acceptor_id in the products table -- join
    # on canonical SMILES instead (same scheme dft_sp_cross_from_geom.py uses).
    def canon(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m, canonical=True) if m is not None else None

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

    # Interaction / complementarity block.
    df["interaction_gap_HOMOd_LUMOa"] = df["donor_xtb_HOMO"] - df["acceptor_xtb_LUMO"]
    df["interaction_fukui_match"] = df["donor_fukui_minus_CHO_C"] * df["acceptor_fukui_plus_CHO_C"]
    for feat in MISMATCH_PAIRS:
        d, a = f"donor_{feat}", f"acceptor_{feat}"
        if d in df.columns and a in df.columns:
            df[f"interaction_absdiff_{feat}"] = (df[d] - df[a]).abs()

    df["pair_key"] = df.apply(lambda r: "__".join(sorted([r.donor_id, r.acceptor_id])), axis=1)

    keep_meta = ["id", "donor_id", "acceptor_id", "pair_key", "reaction_type",
                 "donor_smiles", "acceptor_smiles", "smiles",
                 "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"]
    feat_cols = ([f"donor_{c}" for c in ALDEHYDE_FEATS] +
                 [f"acceptor_{c}" for c in ALDEHYDE_FEATS] +
                 [c for c in PRODUCT_FEATS if c in df.columns] +
                 list(donor2d.columns) + list(acc2d.columns) + list(prod2d.columns) +
                 [c for c in df.columns if c.startswith("interaction_")])
    feat_cols = list(dict.fromkeys(feat_cols))  # de-dup, keep order
    out = df[keep_meta + feat_cols].copy()

    out.to_parquet(OUT_PARQUET, index=False)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out)} rows x {len(feat_cols)} features -> {OUT_PARQUET}")
    print(f"  unordered pairs: {out['pair_key'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
