#!/usr/bin/env python3
"""
Track B (homo+cross unification) — prepare a stratified subsample of the
219k-row homo (self-condensation) library in a format
`assemble_cross_training_table_combined.py`'s `load_round()` can consume
directly, so it can be merged with the cross rounds' training table.

Why this is low-effort: `homo_v6/products_all.csv` was built by the SAME
`cb_featurize.py` used for every cross round (its `--homo-from` mode is
literally "cross pairs where donor==acceptor"), so its column schema
already matches -- donor_id==acceptor_id, all PROD_QM/ALD fields,
dG_gxtb_kcal, computed the identical way. Two gaps closed here:

1. `reaction_type` in products_all.csv is the literal string "homo" for
   every row (cb_featurize's own `FP.reaction_type()` short-circuits to that
   for any is_homo pair) -- recomputed per-row from the aldehyde's own
   `FP.classify()` as e.g. "carbo-carbo"/"hetero-hetero"/"aliph-aliph" so
   these rows land in the SAME reaction_type buckets a cross pair between
   two same-category (but different) aldehydes would, instead of a separate
   meaningless "homo" bucket.
2. Product-side BDE: homo_v6/products_bde_descriptors.csv already has this
   for 219,022 rows as `bde_prod_CC_kcal` (an EARLIER session's compute,
   confirmed as a real full-library gain -- see memory
   [[bde-descriptor-idea]]) -- just needs renaming to `bde_gxtb_kcal` to
   match load_round()'s expected column name. No new compute needed.

DFT labels: `data/raw/dft_sp_funnelv3/dft_labels_all.parquet` (219,364
valid r2SCAN-3c/CPCM-DMSO labels) -- confirmed SAME method/basis/solvent
recipe as cross's `dft_sp_cross_from_geom.py` (`pipeline/compute/
dft_sp_from_geom.py` shares identical --method/--basis/--solvent defaults).

Stratified sampling: equal N per category (aliph-aliph / carbo-carbo /
hetero-hetero) by default, matching candidates_v3's own equal-per-category
convention, so a first joint-training attempt doesn't just drown the
smaller categories in a wash of one dominant class.

Usage
  python cross_benzoin/build_homo_for_unification.py --per-class 10000 \
      --out data/cross_benzoin/homo_unify/homo_unify_v1
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
import featurize_product as FP  # noqa: E402

HOMO_DIR = REPO / "data/cross_benzoin/homo_v6"
PRODUCTS_ALL = HOMO_DIR / "products_all.csv"
PRODUCT_BDE = HOMO_DIR / "products_bde_descriptors.csv"
DFT_LABELS = REPO / "data/raw/dft_sp_funnelv3/dft_labels_all.parquet"

SHORT = {"aliphatic": "aliph", "aromatic_carbo": "carbo", "aromatic_hetero": "hetero"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10000,
                    help="target rows per same-category bucket (x3 = total rows)")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--out", required=True, type=Path,
                    help="output dir; writes {out}_products.csv, {out}_dft.csv, {out}_bde.csv")
    args = ap.parse_args()

    prod = pd.read_csv(PRODUCTS_ALL, low_memory=False)
    prod = prod[prod["error"].astype("string").fillna("") == ""].copy()
    print(f"homo products (error-free): {len(prod)}/{len(prod)}")

    dft = pd.read_parquet(DFT_LABELS, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"])
    dft["id"] = dft["id"].astype(str)
    prod["id"] = prod["id"].astype(str)
    prod = prod.merge(dft, on="id", how="inner")
    print(f"  with DFT label: {len(prod)}")

    bde = pd.read_csv(PRODUCT_BDE, usecols=["id", "bde_prod_CC_kcal"])
    bde["id"] = bde["id"].astype(str)
    bde = bde.rename(columns={"bde_prod_CC_kcal": "bde_gxtb_kcal"}).drop_duplicates("id")
    bde.loc[bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = np.nan
    prod = prod.merge(bde, on="id", how="left")
    print(f"  product BDE coverage: {prod['bde_gxtb_kcal'].notna().sum()}/{len(prod)}")

    # Recompute reaction_type from the aldehyde's own class (donor==acceptor for homo).
    cache: dict[str, str] = {}
    def _cls(smi):
        if smi not in cache:
            cache[smi] = FP.classify(smi)
        return cache[smi]
    ald_cls = prod["donor_smiles"].map(_cls)
    prod["reaction_type"] = ald_cls.map(lambda c: f"{SHORT.get(c,'unk')}-{SHORT.get(c,'unk')}")
    print("  reaction_type (recomputed) counts:\n", prod["reaction_type"].value_counts().to_string())

    # Stratified sample: equal N per same-category bucket.
    rng = np.random.default_rng(args.seed)
    parts = []
    for cat, sub in prod.groupby("reaction_type"):
        n = min(args.per_class, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        parts.append(prod.loc[idx])
    sample = pd.concat(parts, ignore_index=True)
    print(f"\nsampled {len(sample)} rows total")
    print(sample["reaction_type"].value_counts().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    products_out = Path(f"{args.out}_products.csv")
    dft_out = Path(f"{args.out}_dft.csv")
    bde_out = Path(f"{args.out}_bde.csv")

    sample.drop(columns=["dG_orca_kcal", "bde_gxtb_kcal"]).to_csv(products_out, index=False)
    sample[["id", "dG_orca_kcal"]].to_csv(dft_out, index=False)
    sample[["id", "bde_gxtb_kcal"]].to_csv(bde_out, index=False)
    print(f"\nwrote {products_out}\n      {dft_out}\n      {bde_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
