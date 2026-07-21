#!/usr/bin/env python
"""Phase-1 next-step #3, hyperparameter-tuning sub-item (2026-07-15): H-SPOC
(train_local3d_baseline.py) is the strongest cheap CPU baseline (R^2 0.65-0.74,
near-zero marginal compute -- reuses descriptors already on disk) and, unlike B4/B5,
today's XGB config was never tuned at all (plain defaults from `XGBRegressor(...)`
literal in the script). This runs a molecule-grouped random search on the TRAIN split
only (GroupKFold on the same id column the cold split uses, so no test leakage), then
refits the best config and scores it once on the untouched test split next to the
existing default-config number.

Deliberately scoped to H-SPOC only, not B4/B5: those are GPU D-MPNN trainings where even
a small grid means several more multi-hour A100 jobs, and B5 already hit its time limit
once at default settings -- a proper GNN sweep is a bigger, separate resource decision.

Usage:
  python tune_hspoc_xgb.py --which aldehydes --out /tmp/tune_ald.json
  python tune_hspoc_xgb.py --which products  --out /tmp/tune_prod.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

from qc import qc_filter
from splits import molecule_cold_split
from train_local3d_baseline import LOCAL_FEATURES

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

DEFAULT_PARAMS = dict(n_estimators=600, max_depth=4, learning_rate=0.03,
                       subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0)

PARAM_DIST = dict(
    n_estimators=randint(200, 1200),
    max_depth=randint(3, 8),
    learning_rate=loguniform(1e-2, 3e-1),
    subsample=[0.6, 0.7, 0.8, 0.9, 1.0],
    colsample_bytree=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    reg_lambda=loguniform(0.1, 10.0),
    min_child_weight=randint(1, 10),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--target", choices=["bde", "bdfe"], default="bde")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--cv-folds", type=int, default=4)
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
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)

    split = molecule_cold_split(df[id_col], test_frac=args.test_frac, seed=args.seed)
    tr, te = (split == "train").to_numpy(), (split == "test").to_numpy()
    groups_tr = df.loc[tr, id_col].to_numpy()
    print(f"{args.which}: train={tr.sum()} test={te.sum()} unique train groups="
          f"{len(np.unique(groups_tr))}", flush=True)

    default_model = XGBRegressor(random_state=args.seed, n_jobs=-1, **DEFAULT_PARAMS)
    default_model.fit(X[tr], y[tr])
    default_pred = default_model.predict(X[te])

    gkf = GroupKFold(n_splits=args.cv_folds)
    cv_splits = list(gkf.split(X[tr], y[tr], groups=groups_tr))
    search = RandomizedSearchCV(
        XGBRegressor(random_state=args.seed, n_jobs=-1), PARAM_DIST,
        n_iter=args.n_iter, cv=cv_splits,
        scoring="neg_mean_absolute_error", random_state=args.seed, n_jobs=1, verbose=1)
    search.fit(X[tr], y[tr])
    tuned_pred = search.best_estimator_.predict(X[te])

    result = {
        "which": args.which, "target": ycol, "n": len(df),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "default_params": DEFAULT_PARAMS,
        "default": {
            "MAE": float(mean_absolute_error(y[te], default_pred)),
            "RMSE": float(root_mean_squared_error(y[te], default_pred)),
            "R2": float(r2_score(y[te], default_pred)),
            "spearman_rho": float(spearmanr(y[te], default_pred).correlation),
        },
        "tuned_params": search.best_params_,
        "tuned_cv_neg_MAE": float(search.best_score_),
        "tuned": {
            "MAE": float(mean_absolute_error(y[te], tuned_pred)),
            "RMSE": float(root_mean_squared_error(y[te], tuned_pred)),
            "R2": float(r2_score(y[te], tuned_pred)),
            "spearman_rho": float(spearmanr(y[te], tuned_pred).correlation),
        },
    }
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
