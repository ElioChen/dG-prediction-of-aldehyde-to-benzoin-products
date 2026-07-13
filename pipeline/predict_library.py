#!/usr/bin/env python
"""Batch ΔG prediction for the full cross-benzoin homo library (g-xTB Δ-model).

Applies a Δ-model trained by build_cb_training_table.py + sweep_delta.py (feature set =
`ald_*` aldehyde QM + `prod_*` product QM + `dG_xtb_kcal` baseline; here the baseline IS
g-xTB) to the precomputed 220k features — no per-molecule recompute.

    dG_pred = dG_gxtb_kcal + model(features)        # g-xTB baseline + DFT correction

Schema is reconstructed to match training: aldehyde columns prefixed `ald_`, product
columns prefixed `prod_`, and the model's `dG_xtb_kcal` feature is fed the g-xTB value.
Join is by id (products.donor_id == aldehydes.id) — fast, no canonicalization.

  python pipeline/predict_library.py --models-dir runs_cb_gxtb/models \
      --out data/analysis/library_homo_v6_dG_pred_gxtb.csv
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/scratch-shared/schen3/benzoin-dg")
OUT = REPO / "data/cross_benzoin/homo_v6"
ALD_DROP = {"id", "smiles", "xtb_optimized", "error", "xyz_file", "G_xtb", "G_gxtb"}
PROD_DROP = {"id", "donor_id", "acceptor_id", "donor_smiles", "acceptor_smiles", "smiles",
             "reaction_type", "is_homo", "xtb_optimized", "error", "xyz_file",
             "G_donor", "G_acceptor", "G_xtb", "G_donor_gxtb", "G_acceptor_gxtb", "G_gxtb",
             "dG_xtb_kcal", "dG_gxtb_kcal"}


def _num_prefixed(df, drop, prefix):
    cols = [c for c in df.columns if c not in drop and not c.startswith(("adch_", "qtaim_"))
            and pd.api.types.is_numeric_dtype(df[c])]
    return df[cols].add_prefix(prefix), cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(REPO / "runs_cb_gxtb/models"))
    ap.add_argument("--baseline-col", default="dG_gxtb_kcal", help="g-xTB baseline in products")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import joblib
    md = Path(args.models_dir)
    model = joblib.load(md / "delta_model.joblib")
    feats = json.loads((md / "feature_list.json").read_text())
    medians = json.loads((md / "metadata.json").read_text()).get("feature_medians", {})
    print(f"model feats={len(feats)}  baseline_col={args.baseline_col}")

    ald = pd.read_csv(OUT / "aldehydes_all.csv", low_memory=False)
    prod = pd.read_csv(OUT / "products_all.csv", low_memory=False)
    ald_f, _ = _num_prefixed(ald, ALD_DROP, "ald_"); ald_f["id"] = ald["id"]
    prod_f, _ = _num_prefixed(prod, PROD_DROP, "prod_")
    base = pd.to_numeric(prod[args.baseline_col], errors="coerce")

    df = pd.concat([prod[["id", "donor_id", "smiles", "error"]], prod_f,
                    base.rename("dG_xtb_kcal")], axis=1)
    df = df.merge(ald_f, left_on="donor_id", right_on="id", how="left", suffixes=("", "_a"))

    for f in feats:
        if f not in df.columns:
            df[f] = np.nan
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    for f in feats:
        X[f] = X[f].fillna(medians.get(f, X[f].median()))

    df["dG_correction"] = model.predict(X)
    df["dG_gxtb_baseline"] = base.values
    df["dG_pred"] = df["dG_gxtb_baseline"] + df["dG_correction"]
    df["favorable"] = df["dG_pred"] < 0

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["id", "donor_id", "smiles", "dG_gxtb_baseline", "dG_correction", "dG_pred", "favorable"]
    df[[c for c in keep if c in df.columns]].to_csv(out, index=False)
    ok = df["dG_pred"].notna().sum()
    print(f"wrote {out}  rows={len(df)} predicted={ok}")
    print(f"dG_pred mean={df['dG_pred'].mean():.2f} std={df['dG_pred'].std():.2f} "
          f"favorable={int(df['favorable'].sum())} ({100*df['favorable'].mean():.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
