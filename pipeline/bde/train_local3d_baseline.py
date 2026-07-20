#!/usr/bin/env python
"""Phase-1 baseline "H-SPOC" (2026-07-15, user-suggested 3D-structure variant): predict
BDE directly from this project's own LOCAL, 3D-geometry-derived QM descriptors at the
reacting atoms -- Fukui indices, Wiberg bond orders, ADCH/QTAIM charges, vbur, sterimol --
already computed (xtb + Multiwfn) for the entire library in aldehydes_all.csv/
products_all.csv. No fragment generation, no ALFABET dependency, no new compute at all --
this is a straight single-molecule regression on columns that already exist.

Motivation: B3 (train_dspoc_baseline.py) showed that GLOBAL 2D descriptor differences
degenerate to near-zero for the aldehyde's asymmetric fragment pair (acyl radical + lone
H) -- the real chemistry lives in LOCAL electronic structure at the breaking bond, which
global whole-molecule descriptors (even a rich 217-feature RDKit set) can't see directly.
This project already computed exactly that local signal at 3D-optimized geometry level
(fukui_plus_CHO_C, wbo_CO, adch_CHO_C, ...) for the *intact* parent molecule -- using it
tests whether "local 3D reactivity indices at the bond" beats "global 2D differencing"
head-to-head, on the SAME target and split.

Usage:
  python train_local3d_baseline.py --which aldehydes --out /tmp/local3d_ald.json
  python train_local3d_baseline.py --which products  --out /tmp/local3d_prod.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from qc import qc_filter
from splits import molecule_cold_split

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

GLOBAL_XTB = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta",
              "xtb_omega", "xtb_dipole"]

LOCAL_FEATURES = {
    "aldehydes": GLOBAL_XTB + [
        "mulliken_CHO_C", "mulliken_CHO_O", "fukui_plus_CHO_C", "fukui_minus_CHO_C",
        "fukui_0_CHO_C", "dual_descriptor_CHO_C", "wbo_CO", "pa_CHO_O", "vbur_CHO_C",
        "sterimol_L", "sterimol_B1", "sterimol_B5", "SASA_total", "P_int",
        "adch_CHO_C", "adch_CHO_O", "adch_fukui_plus_CHO_C", "adch_fukui_minus_CHO_C",
        "adch_fukui_minus_CHO_O", "qtaim_lap_CO", "qtaim_ell_CO",
    ],
    "products": GLOBAL_XTB + [
        "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO", "mulliken_hydH",
        "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb", "fukui_plus_ketC", "fukui_minus_ketC",
        "fukui_0_ketC", "dual_ketC", "fukui_plus_carbC", "fukui_minus_carbC", "fukui_0_carbC",
        "dual_carbC", "pa_ketO", "vbur_ketC", "vbur_carbC", "sterimol_L", "sterimol_B1",
        "sterimol_B5", "SASA_total", "P_int", "hb_dist", "hb_angle", "dih_core",
        "adch_ketC", "adch_ketO", "adch_carbC", "adch_hydO", "adch_hydH",
        "adch_fukui_plus_ketC", "adch_fukui_minus_ketC", "adch_fukui_plus_carbC",
        "adch_fukui_minus_carbC", "qtaim_rho_CO_ket", "qtaim_lap_CO_ket", "qtaim_ell_CO_ket",
        "qtaim_rho_CC_new", "qtaim_lap_CC_new", "qtaim_ell_CC_new", "qtaim_rho_HB",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--target", choices=["bde", "bdfe"], default="bde")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    feats = LOCAL_FEATURES[args.which]
    id_col = "id" if args.which == "aldehydes" else "donor_id"

    ald = pd.read_csv(H / f"{args.which}_all.csv",
                       usecols=["id", id_col, "error"] + feats if id_col != "id"
                       else ["id", "error"] + feats,
                       dtype={"id": str, id_col: str} if id_col != "id" else {"id": str},
                       keep_default_na=False, low_memory=False)
    ald = ald[ald["error"] == ""]
    for c in feats:
        ald[c] = pd.to_numeric(ald[c], errors="coerce")

    labels = pd.read_csv(H / f"{args.which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    ycol = f"{args.target}_gxtb_kcal"
    labels = labels.dropna(subset=[ycol]).drop_duplicates("id")
    labels = labels[qc_filter(labels[ycol])]

    df = labels.merge(ald, on="id", how="inner").dropna(subset=feats, how="all")
    print(f"{args.which}: {len(df)} rows with BDE label + local-3D descriptors "
          f"({len(feats)} features)", flush=True)

    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)

    split = molecule_cold_split(df[id_col], test_frac=args.test_frac, seed=args.seed)
    tr, te = (split == "train").to_numpy(), (split == "test").to_numpy()
    print(f"train={tr.sum()}  test={te.sum()}  cold on '{id_col}'", flush=True)

    model = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                          random_state=args.seed, n_jobs=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    mean_pred = np.full(te.sum(), y[tr].mean())

    importances = dict(sorted(zip(feats, model.feature_importances_.tolist()),
                               key=lambda kv: -kv[1]))
    result = {
        "which": args.which, "target": ycol, "n": len(df), "n_features": len(feats),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "model": {
            "MAE": float(mean_absolute_error(y[te], pred)),
            "RMSE": float(root_mean_squared_error(y[te], pred)),
            "R2": float(r2_score(y[te], pred)),
            "spearman_rho": float(spearmanr(y[te], pred).correlation),
        },
        "mean_baseline_MAE": float(mean_absolute_error(y[te], mean_pred)),
        "feature_importance": importances,
    }
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
