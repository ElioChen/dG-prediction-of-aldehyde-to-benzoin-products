#!/usr/bin/env python
"""Build the TRUE cross-benzoin (donor != acceptor) Δ-learning training table.

Unlike build_cb_training_table.py (homo-only: one aldehyde keyed by donor_smiles,
prefix `ald_`/`prod_`), this joins TWO distinct aldehyde feature blocks -- donor and
acceptor -- plus the product's own features, following the role-aware descriptor
policy in cross_benzoin/docs/DESCRIPTOR_POLICY_CROSS.md:

  donor_*        aldehyde QM descriptors of the donor (nucleophile side)
  acceptor_*     aldehyde QM descriptors of the acceptor (electrophile side)
  product_*      the directed product's own QM descriptors
  interaction_*  donor-vs-acceptor same-descriptor differences (both are aldehydes,
                 so they share an identical raw schema), plus the two named
                 cross-terms the policy doc calls out explicitly (frontier-orbital
                 gap, Fukui pairing)

Target: dG_orca_kcal (DFT, from pipeline/compute/dft_sp_cross_from_geom.py's output).
Δ-baseline: dG_xtb_kcal or dG_gxtb_kcal (GFN2 or g-xTB), matching the existing
delta_core convention of predicting (DFT - baseline), not raw DFT.

Usage:
  python build_cb_cross_training_table.py \
      --products-csv data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_products.csv \
      --dft-labels data/raw/dft_sp_cross/cross_pilot_v1_dft_sp.csv \
      --baseline gxtb --out data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_train_gxtb.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

REPO = Path("/scratch-shared/schen3/benzoin-dg")
ALD_ALL = REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv"
BASELINE_COL = {"gfn2": "dG_xtb_kcal", "gxtb": "dG_gxtb_kcal"}

ALD_DROP = {"id", "smiles", "xtb_optimized", "error", "xyz_file", "G_xtb", "G_gxtb"}
PROD_DROP = {"id", "donor_id", "acceptor_id", "donor_smiles", "acceptor_smiles", "smiles",
             "reaction_type", "is_homo", "xtb_optimized", "error", "xyz_file",
             "G_donor", "G_acceptor", "G_xtb", "G_donor_gxtb", "G_acceptor_gxtb", "G_gxtb",
             "dG_xtb_kcal", "dG_gxtb_kcal"}


def canon(s):
    m = Chem.MolFromSmiles(str(s)) if s else None
    return Chem.MolToSmiles(m) if m else None


def _numeric_feats(df: pd.DataFrame, drop: set[str]) -> list[str]:
    keep = []
    for c in df.columns:
        if c in drop or c.startswith(("adch_", "qtaim_")):   # Multiwfn: empty at this scale
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            keep.append(c)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products-csv", required=True)
    ap.add_argument("--dft-labels", required=True,
                    help="CSV with columns id,dG_orca_kcal (id = donor_id__acceptor_id)")
    ap.add_argument("--aldehydes-all", default=str(ALD_ALL))
    ap.add_argument("--baseline", choices=["gfn2", "gxtb"], default="gxtb")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base_col = BASELINE_COL[args.baseline]

    ald = pd.read_csv(args.aldehydes_all, low_memory=False)
    ald["c"] = ald["smiles"].map(canon)
    ald = ald.dropna(subset=["c"]).drop_duplicates("c")
    ald_cols = _numeric_feats(ald, ALD_DROP)
    ald_idx = ald.set_index("c")[ald_cols]
    print(f"aldehyde QM features: {len(ald_cols)}  (from {len(ald_idx)} cached aldehydes)")

    prod = pd.read_csv(args.products_csv, low_memory=False)
    prod = prod[prod["error"].fillna("") == ""].copy()
    prod["donor_c"] = prod["donor_smiles"].map(canon)
    prod["acceptor_c"] = prod["acceptor_smiles"].map(canon)
    prod = prod.dropna(subset=["donor_c", "acceptor_c"])
    prod_cols = _numeric_feats(prod, PROD_DROP)
    print(f"product QM features: {len(prod_cols)}  (from {len(prod)} labeled products)")

    lab = pd.read_csv(args.dft_labels, low_memory=False)
    lab = lab[lab["dG_orca_kcal"].notna()][["id", "dG_orca_kcal"]]
    print(f"DFT labels available: {len(lab)}")

    df = prod.merge(lab, on="id", how="inner")
    print(f"products with a DFT label: {len(df)}")

    donor_feat = ald_idx.reindex(df["donor_c"]).add_prefix("donor_").reset_index(drop=True)
    acceptor_feat = ald_idx.reindex(df["acceptor_c"]).add_prefix("acceptor_").reset_index(drop=True)
    product_feat = df[prod_cols].add_prefix("product_").reset_index(drop=True)

    # interaction block: same-descriptor donor-vs-acceptor differences (both aldehyde-shaped,
    # identical raw schema) + the two explicitly named cross-terms from DESCRIPTOR_POLICY_CROSS.md
    inter = pd.DataFrame(index=product_feat.index)
    for c in ald_cols:
        d, a = donor_feat.get(f"donor_{c}"), acceptor_feat.get(f"acceptor_{c}")
        if d is None or a is None:
            continue
        inter[f"interaction_diff_{c}"] = d.to_numpy() - a.to_numpy()
    if "donor_xtb_HOMO" in donor_feat and "acceptor_xtb_LUMO" in acceptor_feat:
        inter["interaction_HOMO_donor_LUMO_acceptor_gap"] = (
            donor_feat["donor_xtb_HOMO"].to_numpy() - acceptor_feat["acceptor_xtb_LUMO"].to_numpy())
    if "donor_fukui_minus_CHO_C" in donor_feat and "acceptor_fukui_plus_CHO_C" in acceptor_feat:
        inter["interaction_fukui_donor_minus_x_acceptor_plus"] = (
            donor_feat["donor_fukui_minus_CHO_C"].to_numpy()
            * acceptor_feat["acceptor_fukui_plus_CHO_C"].to_numpy())
    print(f"interaction features: {inter.shape[1]}")

    out = pd.concat([
        df[["id", "donor_id", "acceptor_id", "smiles", "is_homo"]].reset_index(drop=True),
        df[[base_col, "dG_orca_kcal"]].rename(columns={base_col: "dG_xtb_kcal"}).reset_index(drop=True),
        donor_feat, acceptor_feat, product_feat, inter,
    ], axis=1)
    out = out.dropna(subset=["dG_xtb_kcal", "dG_orca_kcal"])
    n_feat = donor_feat.shape[1] + acceptor_feat.shape[1] + product_feat.shape[1] + inter.shape[1]
    print(f"final table: {len(out)} rows, {n_feat} features "
          f"(donor {donor_feat.shape[1]} + acceptor {acceptor_feat.shape[1]} "
          f"+ product {product_feat.shape[1]} + interaction {inter.shape[1]})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")
    corr = out["dG_orca_kcal"] - out["dG_xtb_kcal"]
    print(f"Δ-target (dG_orca - dG_xtb_baseline): mean={corr.mean():.2f} std={corr.std():.2f} "
          f"min={corr.min():.1f} max={corr.max():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
