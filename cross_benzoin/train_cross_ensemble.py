#!/usr/bin/env python3
"""
Frozen-holdout eval + shippable packaging for the MLP+XGB ensemble architecture
validated (CV-only) in architecture_ensemble_experiment.py -- see
[[cross-round4-5-scaleup-and-architecture-win-20260716]]. That experiment found
a real, scale-growing CV gain (-4.9% MAE at 4120 rows, -9.5% at 17270 rows) but
never got a frozen molecule-disjoint holdout number or a joblib artifact, so it
couldn't be compared to train_cross_delta.py's shipped single-XGB champion
(frozen holdout MAE 2.967-2.983) on the same footing. This script closes both
gaps using the exact same candidates_v3 split-map machinery train_cross_delta.py
uses (pair_split_labels, frozen_holdout_eval semantics), so the two are
apples-to-apples: same held-out test rows, same train/validation exclusion for
the final shipped fit.

Feature set: reuses train_cross_delta.py's "all_raw_blocks+mordred" block (raw
descriptor blocks + mordred, interaction terms excluded) rather than "every
column in the table" -- architecture_ensemble_experiment.py used all columns
(including the 10 interaction_* terms confirmed useless three times over),
which was a minor feature-set mismatch against the single-XGB champion's own
feature selection. Fixed here for a clean comparison.

Usage
  python cross_benzoin/train_cross_ensemble.py \
      --table data/cross_benzoin/cross_round5/cross_train_table_5rounds_mordred_slim120.parquet \
      --folds 5 --repeats 10 \
      --outdir data/cross_benzoin/cross_round5/train_ensemble_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
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
from train_cross_delta import (  # noqa: E402
    TARGET_COL, BASELINE_COL, _feature_blocks, pair_split_labels,
    make_parity, make_residual,
)

TABLE = REPO / "data/cross_benzoin/cross_round5/cross_train_table_5rounds_mordred_slim120.parquet"
OUT = REPO / "data/cross_benzoin/cross_round5/train_ensemble_v1"


class MLPXGBEnsemble:
    """MLP(128,64) + XGB(depth=5) + XGB(depth=7), mean of the three =
    point prediction. The architecture validated in
    architecture_ensemble_experiment.py, packaged for shipping/reuse: a single
    joblib.load() + .predict(df) call handles imputation, MLP scaling, and
    averaging internally. Must stay importable from this module path for
    unpickling (joblib pickles by reference, not by value)."""

    def __init__(self, scaler: StandardScaler, mlp: MLPRegressor,
                 xgb_a: XGBRegressor, xgb_b: XGBRegressor,
                 medians: pd.Series, feats: list[str]):
        self.scaler = scaler
        self.mlp = mlp
        self.xgb_a = xgb_a
        self.xgb_b = xgb_b
        self.medians = medians
        self.feats = feats

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        Xdf = df[self.feats].apply(pd.to_numeric, errors="coerce")
        return Xdf.fillna(self.medians).fillna(0.0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prep(df)
        Xs = self.scaler.transform(X)
        return (self.mlp.predict(Xs) + self.xgb_a.predict(X) + self.xgb_b.predict(X)) / 3.0


def _mlp(seed: int) -> MLPRegressor:
    return MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
                        early_stopping=True, n_iter_no_change=20, random_state=seed)


def _xgb(depth: int, ne: int, seed: int) -> XGBRegressor:
    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.04,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                        random_state=seed, n_jobs=-1)


def _fit_ensemble(X: pd.DataFrame, y: np.ndarray, seed: int):
    sc = StandardScaler().fit(X)
    mlp = _mlp(seed).fit(sc.transform(X), y)
    xa = _xgb(depth=5, ne=400, seed=seed).fit(X, y)
    xb = _xgb(depth=7, ne=400, seed=seed + 1).fit(X, y)
    return sc, mlp, xa, xb


def repeated_group_cv_ensemble(df: pd.DataFrame, feats: list[str], groups: np.ndarray,
                               folds: int, repeats: int, seed: int):
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xdf.median(numeric_only=True)
    X = Xdf.fillna(medians)
    y = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()
    oof_sum = np.zeros(len(df))
    oof_cnt = np.zeros(len(df))
    for r in range(repeats):
        gkf = GroupKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        for tr, te in gkf.split(X, y, groups=groups):
            sc, mlp, xa, xb = _fit_ensemble(X.iloc[tr], y[tr], seed)
            Xte_s = sc.transform(X.iloc[te])
            pred = (mlp.predict(Xte_s) + xa.predict(X.iloc[te]) + xb.predict(X.iloc[te])) / 3.0
            oof_sum[te] += pred
            oof_cnt[te] += 1
    oof = oof_sum / np.maximum(oof_cnt, 1)
    pred_dft = df[BASELINE_COL].to_numpy() + oof
    delta = dc.metrics_vs_dft(df[TARGET_COL].to_numpy(), pred_dft)
    base_gxtb = dc.metrics_vs_dft(df[TARGET_COL].to_numpy(), df[BASELINE_COL].to_numpy())
    base_xtb = dc.metrics_vs_dft(df[TARGET_COL].to_numpy(), df["dG_xtb_kcal"].to_numpy())
    return delta, base_gxtb, base_xtb, pred_dft, medians


def frozen_holdout_eval_ensemble(df: pd.DataFrame, feats: list[str], seed: int):
    """Mirrors train_cross_delta.py's frozen_holdout_eval: fit on candidates_v3's
    'train'-split rows, evaluate once on its frozen 'test'-split rows. Same
    split, same exclusion logic -- directly comparable to the single-XGB
    champion's own frozen-holdout number."""
    pair_split = pair_split_labels(df, verbose=False)
    if pair_split is None:
        return None

    train_mask = (pair_split == "train").to_numpy()
    test_mask = (pair_split == "test").to_numpy()
    if train_mask.sum() < 20 or test_mask.sum() < 5:
        print("  too few rows in train/test split for a frozen holdout fit -- skipped")
        return None

    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xdf[train_mask].median(numeric_only=True)
    X = Xdf.fillna(medians)
    y = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()

    sc, mlp, xa, xb = _fit_ensemble(X[train_mask], y[train_mask], seed)
    Xte_s = sc.transform(X[test_mask])
    pred = (mlp.predict(Xte_s) + xa.predict(X[test_mask]) + xb.predict(X[test_mask])) / 3.0
    pred_dft_test = df[BASELINE_COL].to_numpy()[test_mask] + pred
    delta = dc.metrics_vs_dft(df[TARGET_COL].to_numpy()[test_mask], pred_dft_test)
    base_gxtb = dc.metrics_vs_dft(df[TARGET_COL].to_numpy()[test_mask], df[BASELINE_COL].to_numpy()[test_mask])
    return {"n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "delta": delta, "gxtb_baseline": base_gxtb}


def make_xgb_importance(xgb_a: XGBRegressor, xgb_b: XGBRegressor, feats: list[str], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imp = (xgb_a.feature_importances_ + xgb_b.feature_importances_) / 2.0
    order = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([feats[i] for i in order][::-1], imp[order][::-1])
    ax.set_xlabel("mean XGB gain importance (2-XGB-component average)")
    ax.set_title("MLP+XGB ensemble -- XGB-component feature importance\n"
                 "(MLP contribution not captured by this plot)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.table)
    print(f"loaded {len(df)} rows, {df['pair_key'].nunique()} unordered pairs")

    all_feats = [c for c in df.columns if c not in {
        "id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
        "donor_smiles", "acceptor_smiles", "smiles",
        "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"}]
    feats = _feature_blocks(all_feats)["all_raw_blocks+mordred"]
    groups = df["pair_key"].to_numpy()
    print(f"using {len(feats)} feats (all_raw_blocks+mordred, interaction_* excluded "
          f"for parity with the single-XGB champion's feature selection)")

    print(f"\n=== MLP(128,64)+XGB(d5)+XGB(d7) ensemble ({args.folds}x{args.repeats} grouped CV) ===")
    delta, base_gxtb, base_xtb, pred_dft, _ = repeated_group_cv_ensemble(
        df, feats, groups, args.folds, args.repeats, args.seed)
    print(f"  g-xTB baseline   MAE={base_gxtb['MAE']:6.3f}  RMSE={base_gxtb['RMSE']:6.3f}  R2={base_gxtb['R2']:6.3f}")
    print(f"  xTB baseline     MAE={base_xtb['MAE']:6.3f}  RMSE={base_xtb['RMSE']:6.3f}  R2={base_xtb['R2']:6.3f}")
    print(f"  Ensemble         MAE={delta['MAE']:6.3f}  RMSE={delta['RMSE']:6.3f}  R2={delta['R2']:6.3f}")
    print(f"  MAE improvement over g-xTB: {base_gxtb['MAE'] - delta['MAE']:+.3f} kcal/mol")

    print("\n=== Frozen molecule-disjoint holdout (candidates_v3 train/test split) ===")
    frozen = frozen_holdout_eval_ensemble(df, feats, args.seed)
    if frozen:
        fd, fb = frozen["delta"], frozen["gxtb_baseline"]
        print(f"  n_train={frozen['n_train']}  n_test={frozen['n_test']}")
        print(f"  g-xTB baseline   MAE={fb['MAE']:6.3f}  RMSE={fb['RMSE']:6.3f}  R2={fb['R2']:6.3f}")
        print(f"  Ensemble         MAE={fd['MAE']:6.3f}  RMSE={fd['RMSE']:6.3f}  R2={fd['R2']:6.3f}")

    # Final fit for shipping -- same leakage-safe exclusion as train_cross_delta.py:
    # drop candidates_v3 test/validation rows, keep 'train'-split + unlabeled
    # (legacy round1-2) rows.
    pair_split_final = pair_split_labels(df, verbose=False)
    if pair_split_final is not None:
        leak_mask = pair_split_final.isin(["test", "validation"]).to_numpy()
        if leak_mask.any():
            print(f"\nExcluding {leak_mask.sum()} rows in candidates_v3's test/validation split "
                  f"from the final shipped model")
        clean_mask = ~leak_mask
    else:
        clean_mask = np.ones(len(df), dtype=bool)

    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    final_medians = Xdf[clean_mask].median(numeric_only=True)
    X_clean = Xdf[clean_mask].fillna(final_medians)
    y_clean = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()[clean_mask]
    sc, mlp, xa, xb = _fit_ensemble(X_clean, y_clean, args.seed)
    final = MLPXGBEnsemble(sc, mlp, xa, xb, final_medians, feats)

    make_parity(df, pred_dft, base_gxtb, delta, out / "figs" / "parity.png")
    make_residual(df, pred_dft, out / "figs" / "residual_by_category.png")
    make_xgb_importance(xa, xb, feats, out / "figs" / "xgb_component_importance.png")

    cv_csv = out / "data" / "cv_predictions.csv"
    dfo = df[["id", "donor_id", "acceptor_id", "pair_key", "reaction_type",
             BASELINE_COL, "dG_xtb_kcal", TARGET_COL]].copy()
    dfo["dG_pred"] = pred_dft
    dfo["abs_err_ensemble"] = (dfo["dG_pred"] - dfo[TARGET_COL]).abs()
    dfo["abs_err_gxtb"] = (dfo[BASELINE_COL] - dfo[TARGET_COL]).abs()
    dfo.to_csv(cv_csv, index=False)

    by_cat = dfo.groupby("reaction_type")[["abs_err_ensemble", "abs_err_gxtb"]].mean()
    by_cat.to_csv(out / "data" / "mae_by_category.csv")
    print("\nMAE by category pair (ensemble vs g-xTB baseline):")
    print(by_cat.to_string())

    joblib.dump(final, out / "models" / "cross_ensemble_model.joblib")
    (out / "models" / "feature_list.json").write_text(json.dumps(feats, indent=2))
    (out / "models" / "metadata.json").write_text(json.dumps({
        "model": "mlp128_64+xgb_d5+xgb_d7_ensemble", "target": TARGET_COL, "baseline": BASELINE_COL,
        "n_samples": int(len(df)), "n_pairs": int(df["pair_key"].nunique()),
        "n_features": len(feats), "folds": args.folds, "repeats": args.repeats,
        "cv_ensemble": delta, "cv_gxtb_baseline": base_gxtb, "cv_xtb_baseline": base_xtb,
        "frozen_holdout_candidates_v3": frozen,
        "n_final_fit": int(clean_mask.sum()),
        "n_excluded_test_validation_leakage": int((~clean_mask).sum()),
        "feature_medians": {k: float(v) for k, v in final_medians.items()},
    }, indent=2))

    print(f"\nSaved model + figs + metadata to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
