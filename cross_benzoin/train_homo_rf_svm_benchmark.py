#!/usr/bin/env python
"""Mainstream tabular-model benchmark for the homo full-library table
(2026-09-20, user: XGBoost alone isn't enough, compare against other
mainstream models -- random forest, SVM). Same Delta-learning setup, same
394-feature schema, same scaffold-disjoint split as the champion single-XGB
(`train_scaffold_disjoint.py`) -- only the estimator changes, so any MAE
delta is attributable to the model family, not a confound.

Random Forest uses the full training set (scales fine at this size).
SVM does not: RBF-kernel SVR is O(n^2)-O(n^3) and 132k rows is infeasible on
a single node in reasonable time, so it's fit on a random subsample (default
20,000 rows) and reported as such, alongside a LinearSVR fit on the FULL
training set (linear kernel scales like a linear model, no subsampling
needed) -- both numbers, sample size disclosed, not a single hidden-caveat
number.

Usage:
    /home/schen3/venv/nhc-workflow/bin/python cross_benzoin/train_homo_rf_svm_benchmark.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402

TABLE = REPO / "data/cross_benzoin/homo_standalone/homo_standalone_full_library_table_slim260.parquet"
CHAMPION_FEATS = REPO / "data/cross_benzoin/homo_standalone/tabular_full/models/feature_list.json"
OUT = REPO / "data/cross_benzoin/homo_standalone/tabular_benchmark"
TARGET_COL = "dG_orca_kcal"
BASELINE_COL = "dG_b973c_kcal"
SVR_SUBSAMPLE_N = 20000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = json.loads(CHAMPION_FEATS.read_text())
    df = pd.read_parquet(TABLE)
    train_df = df[df["new_scaffold_split"] == "train"].reset_index(drop=True)
    test_df = df[df["new_scaffold_split"] == "test"].reset_index(drop=True)
    print(f"loaded {len(df)} rows -> train={len(train_df)} test={len(test_df)}, {len(feats)} feats")

    medians = train_df[feats].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    Xtr = train_df[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)
    Xte = test_df[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)
    ytr = (train_df[TARGET_COL] - train_df[BASELINE_COL]).to_numpy()
    yte_true = test_df[TARGET_COL].to_numpy()
    base_te = test_df[BASELINE_COL].to_numpy()

    results = {}
    predictions = {"id": test_df["id"].to_numpy(), "dG_true": yte_true, "baseline": base_te}

    # --- Random Forest, full training set ---
    t0 = time.time()
    rf = dc.build_model("rf", {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2}, seed=0)
    rf.fit(Xtr, ytr)
    pred_rf = base_te + rf.predict(Xte)
    m_rf = dc.metrics_vs_dft(yte_true, pred_rf)
    m_rf["train_n"] = len(train_df)
    m_rf["fit_seconds"] = time.time() - t0
    print(f"[Random Forest, n_train={len(train_df)}] MAE={m_rf['MAE']:.3f} R2={m_rf['R2']:.3f} "
          f"({m_rf['fit_seconds']:.0f}s)")
    results["random_forest"] = m_rf
    predictions["pred_random_forest"] = pred_rf
    joblib.dump(rf, OUT / "model_random_forest.joblib")
    imp_rf = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    imp_rf.to_json(OUT / "feature_importance_random_forest.json")

    # --- Linear SVR, full training set (linear kernel scales like a linear model) ---
    from sklearn.svm import LinearSVR
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    scaler = StandardScaler().fit(Xtr)
    lsvr = LinearSVR(C=1.0, max_iter=20000, random_state=0)
    lsvr.fit(scaler.transform(Xtr), ytr)
    pred_lsvr = base_te + lsvr.predict(scaler.transform(Xte))
    m_lsvr = dc.metrics_vs_dft(yte_true, pred_lsvr)
    m_lsvr["train_n"] = len(train_df)
    m_lsvr["fit_seconds"] = time.time() - t0
    print(f"[Linear SVR, n_train={len(train_df)}] MAE={m_lsvr['MAE']:.3f} R2={m_lsvr['R2']:.3f} "
          f"({m_lsvr['fit_seconds']:.0f}s)")
    results["linear_svr"] = m_lsvr
    predictions["pred_linear_svr"] = pred_lsvr
    joblib.dump({"scaler": scaler, "model": lsvr}, OUT / "model_linear_svr.joblib")

    # --- RBF SVR, subsampled training set (full-scale RBF is not tractable here) ---
    from sklearn.svm import SVR
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(Xtr), size=min(SVR_SUBSAMPLE_N, len(Xtr)), replace=False)
    Xtr_sub, ytr_sub = Xtr.iloc[sub_idx], ytr[sub_idx]
    t0 = time.time()
    scaler_rbf = StandardScaler().fit(Xtr_sub)
    rsvr = SVR(kernel="rbf", C=10.0, gamma="scale")
    rsvr.fit(scaler_rbf.transform(Xtr_sub), ytr_sub)
    pred_rsvr = base_te + rsvr.predict(scaler_rbf.transform(Xte))
    m_rsvr = dc.metrics_vs_dft(yte_true, pred_rsvr)
    m_rsvr["train_n"] = int(len(sub_idx))
    m_rsvr["fit_seconds"] = time.time() - t0
    m_rsvr["note"] = f"subsampled to {len(sub_idx)}/{len(Xtr)} training rows -- RBF-kernel SVR is " \
                      f"O(n^2)-O(n^3), the full 132k-row training set is not tractable here"
    print(f"[RBF SVR, n_train={len(sub_idx)} (subsampled)] MAE={m_rsvr['MAE']:.3f} R2={m_rsvr['R2']:.3f} "
          f"({m_rsvr['fit_seconds']:.0f}s)")
    results["rbf_svr_subsampled"] = m_rsvr
    predictions["pred_rbf_svr_subsampled"] = pred_rsvr
    joblib.dump({"scaler": scaler_rbf, "model": rsvr, "train_idx": sub_idx}, OUT / "model_rbf_svr_subsampled.joblib")

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(predictions).to_csv(OUT / "test_predictions.csv", index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
