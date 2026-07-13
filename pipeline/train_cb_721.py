#!/usr/bin/env python
"""Δ-learning training with a 70:20:10 train/val/test split (user-requested protocol).

Unlike delta_core.cv_evaluate (repeated K-fold), this uses a fixed random hold-out:
tune hyperparameters on VAL, refit best on TRAIN+VAL, report final metrics on the
untouched TEST 10%. ALL CHO categories (no aromatic-only scope). Baseline column in the
table is g-xTB (relabelled dG_xtb_kcal); prediction = baseline + model(correction).

  python pipeline/train_cb_721.py --parquet data/featurize_cb_homo_train_gxtb.parquet \
      --trials 120 --outdir runs_cb_gxtb_721
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna, xgboost as xgb

REPO = Path("/scratch-shared/schen3/benzoin-dg")
NON_FEAT = {"c", "SMILES", "index", "PubChem_CID", "dG_orca_kcal", "dG_orca_shermo_kcal",
            "dG_shermo_kcal", "aldehyde_smiles", "error"}


def _metrics(dft_true, dft_pred):
    return {"MAE": float(mean_absolute_error(dft_true, dft_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(dft_true, dft_pred))),
            "R2": float(r2_score(dft_true, dft_pred))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--target", default="dG_orca_kcal")
    ap.add_argument("--baseline", default="dG_xtb_kcal", help="baseline col (g-xTB), also a feature")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df[df[args.target].notna() & df[args.baseline].notna()].copy()
    feats = [c for c in df.columns if c not in NON_FEAT and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    base = df[args.baseline].to_numpy()
    dft = df[args.target].to_numpy()
    y = dft - base                     # the DFT correction the model learns
    print(f"rows={len(df)}  features={len(feats)}  baseline={args.baseline}  "
          f"correction mean={y.mean():.2f} std={y.std():.2f}")

    # 70 / 20 / 10 fixed-seed split
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=args.seed)
    va, te = train_test_split(tmp, test_size=1/3, random_state=args.seed)   # 0.20 / 0.10
    print(f"split: train={len(tr)} val={len(va)} test={len(te)}")

    def fit(params, rows):
        m = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=args.seed, **params)
        m.fit(X.iloc[rows], y[rows]); return m

    def objective(t):
        params = dict(
            n_estimators=t.suggest_int("n_estimators", 400, 1500, step=100),
            max_depth=t.suggest_int("max_depth", 3, 8),
            learning_rate=t.suggest_float("learning_rate", 5e-3, 0.1, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-2, 10, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5, log=True),
            min_child_weight=t.suggest_int("min_child_weight", 1, 8))
        m = fit(params, tr)
        pred = base[va] + m.predict(X.iloc[va])
        return mean_absolute_error(dft[va], pred)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = study.best_params
    print(f"best val MAE={study.best_value:.3f}  params={json.dumps(best)}")

    # refit best on train+val, evaluate on the untouched test
    final = fit(best, np.concatenate([tr, va]))
    pred_te = base[te] + final.predict(X.iloc[te])
    test = _metrics(dft[te], pred_te)
    base_te = _metrics(dft[te], base[te])     # g-xTB baseline alone on test
    val_pred = base[va] + final.predict(X.iloc[va])
    val_m = _metrics(dft[va], val_pred)
    print(f"TEST  Δ-model {test}   |  g-xTB baseline {base_te}")

    out = Path(args.outdir); (out / "models").mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(final, out / "models" / "delta_model.joblib")
    (out / "models" / "feature_list.json").write_text(json.dumps(feats, indent=2))
    (out / "models" / "metadata.json").write_text(json.dumps({
        "model": "xgb", "target": args.target, "baseline": "gxtb_cosmo_dmso",
        "split": "70/20/10 random", "seed": args.seed,
        "n_total": len(df), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_features": len(feats), "best_params": best,
        "val_metrics": val_m, "test_metrics": test, "test_baseline_metrics": base_te,
        "feature_medians": {k: float(v) for k, v in medians.items()},
        "scope": "ALL categories (no aromatic filter)",
    }, indent=2))
    print(f"saved -> {out}/models/  (TEST MAE {test['MAE']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
