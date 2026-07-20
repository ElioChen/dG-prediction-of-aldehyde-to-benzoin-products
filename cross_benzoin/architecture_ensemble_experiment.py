#!/usr/bin/env python3
"""
Architecture experiment for cross (previously flagged, never run): does homo's
winning ensemble architecture -- MLP(512,256,128) + XGB(depth=8) + XGB(depth=10),
mean of the three as the point prediction -- beat cross's current single small
XGB (n_estimators=300, max_depth=3, used throughout train_cross_delta.py)?

Deliberately NOT copying homo's XGB hyperparameters verbatim (n_estimators=800+,
depth 7-10, tuned for 150k+ rows) -- those would almost certainly overfit at
cross's current 4-8k row scale. Uses more modest depths/estimator counts sized
for this data regime, per the explicit caution in
[[cross-round3-and-ensemble72-packaging-20260714]] ("should NOT blindly copy
homo's exact hyperparameters").

Same repeated pair-grouped CV protocol as train_cross_delta.py's
repeated_group_cv (GroupKFold by pair_key, folds x repeats, median-impute)
for apples-to-apples comparison against the already-known single-XGB number.

Usage
  python cross_benzoin/architecture_ensemble_experiment.py \
      --table data/cross_benzoin/cross_round3/cross_train_table_3rounds_mordred.parquet \
      --folds 5 --repeats 10
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402
from train_cross_delta import TARGET_COL, BASELINE_COL  # noqa: E402


def _mlp(seed: int) -> MLPRegressor:
    # sized down from homo's (512,256,128) -- cross has ~15-70x fewer rows
    return MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
                        early_stopping=True, n_iter_no_change=20, random_state=seed)


def _xgb(depth: int, ne: int, seed: int) -> XGBRegressor:
    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.04,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                        random_state=seed, n_jobs=-1)


def ensemble_cv(df: pd.DataFrame, feats: list[str], groups: np.ndarray,
                folds: int, repeats: int, seed: int):
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xdf.median(numeric_only=True)
    X = Xdf.fillna(medians).fillna(0.0)  # all-NaN cols (median itself NaN) -> 0, uninformative anyway
    y = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()
    oof_sum = np.zeros(len(df))
    oof_cnt = np.zeros(len(df))
    for r in range(repeats):
        gkf = GroupKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        for tr, te in gkf.split(X, y, groups=groups):
            sc = StandardScaler().fit(X.iloc[tr])
            Xtr_s, Xte_s = sc.transform(X.iloc[tr]), sc.transform(X.iloc[te])
            mlp = _mlp(seed).fit(Xtr_s, y[tr])
            x8 = _xgb(depth=5, ne=400, seed=seed).fit(X.iloc[tr], y[tr])
            x10 = _xgb(depth=7, ne=400, seed=seed + 1).fit(X.iloc[tr], y[tr])
            pred = (mlp.predict(Xte_s) + x8.predict(X.iloc[te]) + x10.predict(X.iloc[te])) / 3.0
            oof_sum[te] += pred
            oof_cnt[te] += 1
    oof = oof_sum / np.maximum(oof_cnt, 1)
    pred_dft = df[BASELINE_COL].to_numpy() + oof
    delta = dc.metrics_vs_dft(df[TARGET_COL].to_numpy(), pred_dft)
    base_gxtb = dc.metrics_vs_dft(df[TARGET_COL].to_numpy(), df[BASELINE_COL].to_numpy())
    return delta, base_gxtb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    meta = {"id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
            "donor_smiles", "acceptor_smiles", "smiles",
            "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"}
    feats = [c for c in df.columns if c not in meta]
    groups = df["pair_key"].to_numpy()
    print(f"{len(df)} rows, {df['pair_key'].nunique()} pairs, {len(feats)} feats")

    print(f"\n=== MLP(128,64) + XGB(d5) + XGB(d7) ensemble, {args.folds}x{args.repeats} grouped CV ===")
    delta, base_gxtb = ensemble_cv(df, feats, groups, args.folds, args.repeats, args.seed)
    print(f"  MAE={delta['MAE']:.3f} RMSE={delta['RMSE']:.3f} R2={delta['R2']:.3f}  "
          f"(g-xTB baseline MAE={base_gxtb['MAE']:.3f})")
    print("\ncompare against train_cross_delta.py's single-XGB baseline on the SAME table "
          "(run separately, or check its ablations.csv 'all_raw_blocks+mordred' row)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
