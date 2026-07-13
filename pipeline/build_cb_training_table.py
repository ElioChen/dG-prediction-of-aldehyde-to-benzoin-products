#!/usr/bin/env python
"""Build the cross-benzoin Δ-learning training table (product-descriptor era).

Joins the existing DFT labels (dG_orca_kcal) onto the NEW cross-benzoin featurization
(aldehyde QM + product QM descriptors, no RDKit 2D), keyed on canonical aldehyde SMILES.
This is the training table for the current modeling direction (user 2026-06-24: no RDKit,
product descriptors now computed). See [[cb-postfeaturize-autochain]] / [[descriptor-slim-v4]].

Construction choices (documented for review):
  * Feature set = aldehyde QM descriptors (prefix `ald_`) + product QM descriptors
    (prefix `prod_`) + `dG_xtb_kcal`. No RDKit 2D block. Multiwfn (adch_/qtaim_) cols
    are all-empty at full scale and are dropped.
  * Δ-target baseline `dG_xtb_kcal` is taken from the CROSS-BENZOIN run (not the label
    run) so train and the 220k-prediction share the same baseline geometry/method. The
    label vs cross-benzoin dG_xtb agree at bias 0.14 / r 0.93 (conformer-search scatter,
    MAE ~1 kcal); delta_core's MAD QC trims the rare large geometry-mismatch outliers.
  * Output keeps `SMILES` (canonical) so delta_core's scope/reactive filters work.

Output: data/featurize_cb_homo_train.parquet  (non-destructive; new file).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

REPO = Path("/scratch-shared/schen3/benzoin-dg")
OUT = REPO / "data/cross_benzoin/homo_v6"
# --baseline gfn2 -> dG_xtb_kcal ; gxtb -> dG_gxtb_kcal (relabelled to dG_xtb_kcal so the
# existing delta_core pipeline treats it as the XTB_DG baseline unchanged).
BASELINE_COL = {"gfn2": "dG_xtb_kcal", "gxtb": "dG_gxtb_kcal"}

ALD_DROP = {"id", "smiles", "xtb_optimized", "error", "xyz_file", "G_xtb", "G_gxtb"}
PROD_DROP = {"id", "donor_id", "acceptor_id", "donor_smiles", "acceptor_smiles", "smiles",
             "reaction_type", "is_homo", "xtb_optimized", "error", "xyz_file",
             "G_donor", "G_acceptor", "G_xtb", "G_donor_gxtb", "G_acceptor_gxtb", "G_gxtb",
             "dG_xtb_kcal", "dG_gxtb_kcal"}


def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None


def _numeric_feats(df, drop):
    keep = []
    for c in df.columns:
        if c in drop:
            continue
        if c.startswith(("adch_", "qtaim_")):   # empty by design at full scale
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            keep.append(c)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["gfn2", "gxtb"], default="gxtb")
    # authoritative r2SCAN-3c labels (user 2026-06-24): assembled from
    # data/raw/featurize_funnelv3_relabel via data/labels_r2scan_relabel.parquet
    ap.add_argument("--labels", default=str(REPO / "data/labels_r2scan_relabel.parquet"))
    args = ap.parse_args()
    base_col = BASELINE_COL[args.baseline]
    DEST = REPO / (f"data/featurize_cb_homo_train_{args.baseline}.parquet")
    print(f"baseline={args.baseline} (col {base_col}) labels={args.labels} -> {DEST.name}")

    lab = pd.read_parquet(args.labels)
    lab = lab[lab["dG_orca_kcal"].notna()].copy()
    if "c" not in lab.columns:
        lab["c"] = lab["SMILES"].map(canon)
    lab = lab.dropna(subset=["c"]).drop_duplicates("c")
    print(f"labels (r2SCAN-3c): {len(lab)}  [ALL categories, no aromatic filter]")

    ald = pd.read_csv(OUT / "aldehydes_all.csv", low_memory=False)
    ald["c"] = ald["smiles"].map(canon)
    ald = ald.dropna(subset=["c"]).drop_duplicates("c")
    af = _numeric_feats(ald, ALD_DROP)
    ald_feat = ald.set_index("c")[af].add_prefix("ald_")
    print(f"aldehyde QM features: {len(af)}")

    prod = pd.read_csv(OUT / "products_all.csv", low_memory=False)
    prod = prod[prod["error"].isna()].copy()
    prod["c"] = prod["donor_smiles"].map(canon)
    prod = prod.dropna(subset=["c"]).drop_duplicates("c")
    pf = _numeric_feats(prod, PROD_DROP)
    prod_feat = prod.set_index("c")[pf].add_prefix("prod_")
    prod_dg = prod.set_index("c")[base_col].rename("dG_xtb_kcal")  # baseline as XTB_DG
    print(f"product QM features: {len(pf)}")

    df = lab[["c", "SMILES", "dG_orca_kcal"]].set_index("c")
    df = df.join(prod_dg, how="inner").join(ald_feat, how="inner").join(prod_feat, how="inner")
    df = df.dropna(subset=["dG_xtb_kcal", "dG_orca_kcal"]).reset_index(drop=True)
    print(f"joined training rows: {len(df)}  total cols: {df.shape[1]}  "
          f"(features={df.shape[1]-3})")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DEST, index=False)
    print(f"wrote {DEST}")
    # quick target sanity
    corr = df["dG_orca_kcal"] - df["dG_xtb_kcal"]
    print(f"Δ-target (dG_orca - dG_xtb): mean={corr.mean():.2f} std={corr.std():.2f} "
          f"min={corr.min():.1f} max={corr.max():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
