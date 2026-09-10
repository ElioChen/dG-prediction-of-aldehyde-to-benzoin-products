#!/usr/bin/env python3
"""
Manifest for the FAST homo relabel route: DFT single-points on the ARCHIVED
geometries (no conformer search, no Hessian), reusing the stored xTB thermal.

Geometry store: data/cross_benzoin/bde_homo_product_featurize_20260902/
chunk_XXXX/geom.tar.zst  (members  xyz/pNNNNNN.xyz  +  ald_xyz/aNNNNNN.xyz).
`products_all.csv` / `aldehydes_all.csv` `xyz_file` columns give the exact
per-species path; the archive + member are derived from it.

thermal (Eh, reused RRHO):
  product  = G_product - xtb_energy   (products_all.csv)
  aldehyde = G_xtb     - xtb_energy   (aldehydes_all.csv)

One row per error-free homo product with both geometries + both thermals
resolvable. Emits: id, donor_id, prod_smiles, ald_smiles, prod_arc, prod_mem,
ald_arc, ald_mem, thermal_prod, thermal_ald, charge_prod, charge_ald,
label_stored, new_scaffold_split.

Usage
  python cross_benzoin/build_homo_sp_manifest.py \
      --out data/cross_benzoin/homo_standalone/relabel_sp/homo_sp_manifest.parquet
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parent.parent
H = REPO / "data/cross_benzoin/homo_v6"
U = REPO / "data/cross_benzoin/homo_unify"


def _arc(xyz: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(xyz)), "geom.tar.zst")


def _charge(smi) -> int:
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.GetFormalCharge(m) if m is not None else 0
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-exists", action="store_true",
                    help="stat every archive (slow on GPFS); default trusts the path")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    p = pd.read_csv(H / "products_all.csv", low_memory=False,
                    usecols=["id", "donor_id", "smiles", "xyz_file", "error",
                             "xtb_energy", "G_product"])
    p = p[p["error"].astype("string").fillna("") == ""].copy()
    p["id"] = pd.to_numeric(p["id"], errors="coerce")
    p["donor_id"] = pd.to_numeric(p["donor_id"], errors="coerce")
    p["thermal_prod"] = p["G_product"] - p["xtb_energy"]
    p["prod_arc"] = p["xyz_file"].map(_arc)
    p["prod_mem"] = "xyz/" + p["xyz_file"].map(os.path.basename)
    p = p.rename(columns={"smiles": "prod_smiles"})
    print(f"error-free homo products: {len(p)}")

    a = pd.read_csv(H / "aldehydes_all.csv", low_memory=False,
                    usecols=["id", "smiles", "xyz_file", "xtb_energy", "G_xtb"])
    a["id"] = pd.to_numeric(a["id"], errors="coerce")
    a["thermal_ald"] = a["G_xtb"] - a["xtb_energy"]
    a["ald_arc"] = a["xyz_file"].map(_arc)
    a["ald_mem"] = "ald_xyz/" + a["xyz_file"].map(os.path.basename)
    a = a[["id", "smiles", "thermal_ald", "ald_arc", "ald_mem"]].rename(
        columns={"id": "donor_id", "smiles": "ald_smiles"})

    m = p.merge(a, on="donor_id", how="left")
    ok = m["thermal_prod"].notna() & m["thermal_ald"].notna() & m["ald_arc"].notna()
    if args.check_exists:
        arcs = pd.unique(pd.concat([m.loc[ok, "prod_arc"], m.loc[ok, "ald_arc"]]))
        present = {x for x in arcs if os.path.exists(x)}
        ok &= m["prod_arc"].isin(present) & m["ald_arc"].isin(present)
        print(f"archives present: {len(present)}/{len(arcs)}")
    m = m[ok].copy()
    print(f"resolvable rows: {len(m)} (dropped {ok.size - ok.sum()})")

    m["charge_prod"] = m["prod_smiles"].map(_charge)
    m["charge_ald"] = m["ald_smiles"].map(_charge)

    lab = pd.read_csv(U / "homo_unify_v1_dft.csv")
    lab["id"] = pd.to_numeric(lab["id"], errors="coerce")
    m = m.merge(lab.rename(columns={"dG_orca_kcal": "label_stored"}), on="id", how="left")

    sp = pd.read_csv(H / "products_scaffold_split.csv")
    sp["id"] = pd.to_numeric(sp["id"], errors="coerce")
    m = m.merge(sp[["id", "scaffold_split"]].rename(
        columns={"scaffold_split": "new_scaffold_split"}), on="id", how="left")

    # shuffle so array segments are split/label balanced; QC (has label) first
    m["_has_label"] = m["label_stored"].notna().astype(int)
    qc = m[m["_has_label"] == 1].sample(min(2000, int(m["_has_label"].sum())), random_state=47)
    rest = m.drop(qc.index).sample(frac=1.0, random_state=47)
    out = pd.concat([qc, rest], ignore_index=True).drop(columns="_has_label")

    cols = ["id", "donor_id", "prod_smiles", "ald_smiles",
            "prod_arc", "prod_mem", "ald_arc", "ald_mem",
            "thermal_prod", "thermal_ald", "charge_prod", "charge_ald",
            "label_stored", "new_scaffold_split"]
    out = out[cols]
    out.to_parquet(args.out, index=False)
    out.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"\nwrote {len(out)} rows -> {args.out}")
    print("  QC (has stored label):", int(out['label_stored'].notna().sum()))
    print("  split:", out["new_scaffold_split"].value_counts(dropna=False).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
