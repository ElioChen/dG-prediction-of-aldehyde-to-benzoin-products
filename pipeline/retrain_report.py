#!/usr/bin/env python3
"""
One-command retrain + report for the funnel_v3 re-label. Produces, on the cleaned
labels, the full picture asked for after the conformer overhaul:

  1. matched-subset Δ-model CV (OLD vs NEW labels, same molecules) — did the noise
     floor drop, and is the gain in the tail (RMSE/max)?
  2. full NEW-label Δ-model zoo CV (xgb/ridge/gpr)         — the accurate xTB-tier
  3. NEW-label 2D surrogate CV (pure SMILES, no xTB)        — the fast tier
  4. AD stratification of the NEW Δ-model error by CHO category and by BENZOIN-PRODUCT
     flexibility bin (the right axis — the product is always floppier than the aldehyde)

Runnable on the partial parquet during the campaign (directional) or the final one.

Usage
  python pipeline/retrain_report.py                         # uses _partial parquet
  python pipeline/retrain_report.py --new data/featurize_funnelv3.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

sys.path[:0] = ["pipeline", "pipeline/compute",
                str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parent / "compute")]
import delta_core as dc
import thermo_orca as Th
from cho_category import cho_class

REPO = Path(__file__).resolve().parent.parent


def _X(tbl):
    return tbl.X.to_numpy() if hasattr(tbl.X, "to_numpy") else np.asarray(tbl.X)


def delta_cv(tbl, mask=None, kind="xgb", folds=5, reps=4, seed=42, return_err=False):
    X, y, xtb, dft = _X(tbl), tbl.y, tbl.dG_xtb, tbl.dG_dft
    if mask is not None:
        X, y, xtb, dft = X[mask], y[mask], xtb[mask], dft[mask]
    rkf = RepeatedKFold(n_splits=folds, n_repeats=reps, random_state=seed)
    maes, rmses, mx = [], [], []
    per = np.zeros(len(X)); cnt = np.zeros(len(X))
    for tr, te in rkf.split(X):
        m = dc.build_model(kind, seed=seed); m.fit(X[tr], y[tr])
        pred = xtb[te] + m.predict(X[te])
        e = np.abs(pred - dft[te])
        maes.append(e.mean()); rmses.append(np.sqrt((e**2).mean())); mx.append(e.max())
        per[te] += e; cnt[te] += 1
    out = (np.mean(maes), np.std(maes), np.mean(rmses), np.mean(mx))
    return (out, per / np.maximum(cnt, 1)) if return_err else out


def _surrogate_factories():
    """REAL models for the 2D-direct surrogate (scaling for the linear one)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    return {
        "xgb": lambda: XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                    random_state=42, n_jobs=-1),
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


def surrogate_cv(tbl, factory, folds=5, reps=4, seed=42):
    from train_surrogate import FEATURES_2D
    df = tbl.df
    feats = [c for c in FEATURES_2D if c in df.columns]
    X = np.nan_to_num(df[feats].to_numpy(float))
    y = tbl.dG_dft                                           # DIRECT target
    rkf = RepeatedKFold(n_splits=folds, n_repeats=reps, random_state=seed)
    maes, rmses = [], []
    for tr, te in rkf.split(X):
        m = factory(); m.fit(X[tr], y[tr])
        e = np.abs(m.predict(X[te]) - y[te])
        maes.append(e.mean()); rmses.append(np.sqrt((e**2).mean()))
    return np.mean(maes), np.std(maes), np.mean(rmses)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None, help="old parquet (default: delta_core default)")
    ap.add_argument("--new", default=str(REPO / "data/featurize_funnelv3_partial.parquet"))
    ap.add_argument("--kinds", nargs="+", default=["xgb", "ridge", "gpr"])
    a = ap.parse_args()

    old = dc.load_training_table() if a.old is None else dc.load_training_table(parquet=a.old)
    new = dc.load_training_table(parquet=a.new)
    print(f"\nOLD n={len(old.df)}  NEW n={len(new.df)}")

    # 1. matched-subset old vs new
    common = sorted(set(old.df["index"]) & set(new.df["index"]))
    mo = old.df["index"].isin(common).to_numpy()
    mn = new.df["index"].isin(common).to_numpy()
    print(f"\n[1] Matched-subset Δ-model CV (n={len(common)}, same molecules):")
    for tag, tbl, msk in (("old", old, mo), ("new", new, mn)):
        mae, sd, rmse, mx = delta_cv(tbl, msk, kind="xgb")
        print(f"    {tag:3s}: MAE {mae:.3f}±{sd:.3f}  RMSE {rmse:.3f}  max {mx:.2f}")

    # 2. full new-label Δ-model zoo — the REAL zoo from explore_models (proper scaled
    #    ridge / GPR / kernel-ridge). regression_zoo() runs + prints the table itself.
    import explore_models as em
    print(f"\n[2] Full NEW-label Δ-model zoo CV (n={len(new.df)}, DFT-level):")
    em.regression_zoo(new)

    # 3. 2D surrogate — REAL models
    print(f"\n[3] NEW-label 2D surrogate CV (pure SMILES, no xTB):")
    for k, fac in _surrogate_factories().items():
        mae, sd, rmse = surrogate_cv(new, fac)
        print(f"    {k:6s}: MAE {mae:.3f}±{sd:.3f}  RMSE {rmse:.3f}")

    # 4. AD stratification of new Δ-model error
    (_, _, _, _), err = delta_cv(new, kind="xgb", return_err=True)
    d = new.df.copy().reset_index(drop=True)
    d["err"] = err
    d["cat"] = d["SMILES"].map(cho_class)
    prb = []
    for s in d["SMILES"]:
        bz = Th._make_benzoin_smiles(s)
        prb.append(Th._mol_rotbonds(bz) if bz else -1)
    d["prod_rb"] = prb
    def fb(rb): return ("rigid_0-3" if rb <= 3 else "mid_4-7" if rb <= 7 else
                        "floppy_8-12" if rb <= 12 else "very_13+")
    d["pflex"] = d["prod_rb"].map(fb)
    print(f"\n[4] NEW Δ-model CV error (MAE) by CHO category:")
    print(d.groupby("cat")["err"].agg(["count", "mean"]).round(3).to_string())
    print(f"\n    by BENZOIN-PRODUCT flexibility bin:")
    print(d.groupby("pflex")["err"].agg(["count", "mean"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
