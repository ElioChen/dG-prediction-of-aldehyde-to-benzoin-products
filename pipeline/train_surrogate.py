#!/usr/bin/env python3
"""
Pure-SMILES 2D surrogate for benzoin ΔG — the FAST, no-quantum tier of the hierarchy.

Where the Δ-model predicts `dG_orca − dG_xtb` from ~62 descriptors (incl. dG_xtb) and so
needs an xTB run per SMILES, this surrogate predicts **dG_orca directly** from only the
geometry-free RDKit-2D descriptors (`calc_rdkit`), i.e. straight from the SMILES string —
no conformer search, no xTB, instant. Lower accuracy, but usable as a pre-filter / to
score 100k+ candidates, and later distillable from the Δ-model's predictions.

Trains on the SAME molecule set as the Δ-model: it calls `delta_core.load_training_table`
(joint aromatic scope = CHO_SCOPE, drop_reactive + QC) so the hierarchy comparison is
apples-to-apples, then swaps in (X = 2D features only, y = dG_orca direct) and runs the
same RepeatedKFold CV with the same seed/folds.

Usage
  python pipeline/train_surrogate.py                       # current parquet, xgb+ridge+gpr
  python pipeline/train_surrogate.py --parquet data/featurize_funnelv3.parquet --kinds xgb ridge
  python pipeline/train_surrogate.py --save                # ship surrogate_model.joblib (xgb)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

import delta_core as dc

# Geometry-free RDKit-2D descriptors (calc_rdkit) — computable from the SMILES alone.
FEATURES_2D = [
    "MW", "ExactMW", "LogP", "MolMR", "TPSA", "HBD", "HBA", "RotBonds",
    "ArRings", "ArHetRings", "AlRings", "Rings", "Heteroatoms", "FractionCSP3",
    "BertzCT", "Chi0v", "Chi1v", "Kappa1", "Kappa2", "LabuteASA",
    "NumStereocenters", "n_CHO",
    "gasteiger_CHO_C", "gasteiger_CHO_O", "gasteiger_maxpos", "gasteiger_minneg",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None,
                    help="featurize parquet (default: delta_core's DEFAULT_FEATURIZE_PARQUET)")
    ap.add_argument("--kinds", nargs="+", default=["xgb", "ridge", "gpr"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true",
                    help="fit the --ship-kind model on ALL data and ship the artifact "
                         "(surrogate_model.joblib + surrogate_features.json + "
                         "surrogate_metadata.json) into --dest")
    ap.add_argument("--ship-kind", default="xgb",
                    help="which model to ship when --save (default: xgb)")
    ap.add_argument("--dest", default=str(dc.REPO_ROOT / "src/benzoin_dG/models"),
                    help="destination package models dir for the shipped surrogate")
    a = ap.parse_args()

    # Same filtered/QC'd, joint-aromatic table the Δ-model trains on.
    kw = {"parquet": a.parquet} if a.parquet else {}
    tbl = dc.load_training_table(**kw)
    df = tbl.df
    feats = [c for c in FEATURES_2D if c in df.columns]
    missing = [c for c in FEATURES_2D if c not in df.columns]
    if missing:
        print(f"WARN missing 2D cols (skipped): {missing}")
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = {c: float(Xdf[c].median()) for c in feats}   # per-feature, for inference impute
    X = Xdf.fillna(pd.Series(medians)).to_numpy(dtype=float)
    y_direct = tbl.dG_dft                                # DIRECT target = DFT ΔG
    print(f"surrogate: n={len(df)} molecules, {len(feats)} 2D features, "
          f"target=dG_orca (direct)\n")

    # Reference: the trivial dG_xtb-only baseline (no ML) and the Δ-model, on this set.
    base = dc.metrics_vs_dft(tbl.dG_dft, tbl.dG_xtb)
    print(f"  [ref] xTB-only (no ML)      MAE {base['MAE']:.3f}  RMSE {base['RMSE']:.3f}")

    # REAL models (delta_core.build_model only knows xgb/rf — everything else silently
    # falls through to GradientBoosting, so construct ridge/gpr explicitly here).
    def factory(kind):
        if kind == "xgb":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                random_state=a.seed, n_jobs=-1)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if kind == "ridge":
            from sklearn.linear_model import Ridge
            return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        if kind == "gpr":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
            k = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
            return make_pipeline(StandardScaler(), GaussianProcessRegressor(
                kernel=k, alpha=1e-6, normalize_y=True))
        raise ValueError(kind)

    rkf = RepeatedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=a.seed)
    cv = {}
    kinds = list(dict.fromkeys(a.kinds + ([a.ship_kind] if a.save else [])))
    for kind in kinds:
        maes, rmses, maxes = [], [], []
        for tr, te in rkf.split(X):
            m = factory(kind)
            m.fit(X[tr], y_direct[tr])
            p = m.predict(X[te])
            d = np.abs(p - y_direct[te])
            maes.append(d.mean()); rmses.append(np.sqrt((d**2).mean())); maxes.append(d.max())
        cv[kind] = dict(MAE=float(np.mean(maes)), RMSE=float(np.mean(rmses)),
                        max=float(np.mean(maxes)))
        print(f"  2D-direct {kind:6s}  MAE {np.mean(maes):.3f}±{np.std(maes):.3f}  "
              f"RMSE {np.mean(rmses):.3f}  max {np.mean(maxes):.2f}")

    print("\n(compare MAE to the Δ-model from explore_models.py / train_delta.py on the "
          "same parquet — that gap is the accuracy cost of dropping the xTB tier.)")

    if a.save:
        dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
        final = factory(a.ship_kind)
        final.fit(X, y_direct)                           # final fit on ALL data
        joblib.dump(final, dest / "surrogate_model.joblib")
        (dest / "surrogate_features.json").write_text(json.dumps(feats, indent=2))
        (dest / "surrogate_metadata.json").write_text(json.dumps({
            "model": a.ship_kind, "target": "dG_orca_kcal (direct)",
            "tier": "2d_surrogate", "n_samples": int(len(df)),
            "n_features": len(feats), "features": feats,
            "feature_medians": medians,
            "cv": cv.get(a.ship_kind), "cv_all": cv,
            "folds": a.folds, "repeats": a.repeats, "seed": a.seed,
            "note": "Pure-SMILES RDKit-2D surrogate (no xTB); predicts DFT dG_orca "
                    "directly. Fast/no-quantum tier of the prediction hierarchy.",
        }, indent=2))
        print(f"\nShipped 2D surrogate ({a.ship_kind}) to {dest}:")
        print(f"  surrogate_model.joblib  surrogate_features.json  surrogate_metadata.json")
        print(f"  n={len(df)}  feats={len(feats)}  CV MAE={cv[a.ship_kind]['MAE']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
