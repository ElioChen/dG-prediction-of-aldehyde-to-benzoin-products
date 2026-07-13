#!/usr/bin/env python3
"""
Broad model exploration for predicting benzoin ΔG (dG_orca_kcal), beyond the
production xgb Δ-learning tree. Three complementary framings, all on the same
repeated-K-fold CV and the same 63-descriptor table (delta_core):

  1. REGRESSION ZOO   — point prediction of the Δ-correction across many model
     families (boosting, bagging, linear, kernel, neighbours, MLP, GP, stacking).
  2. UNCERTAINTY      — probabilistic regressors that emit a predictive *variance*
     (NGBoost, Gaussian Process, quantile gradient boosting). Reported with NLL and
     80% prediction-interval coverage — the basis for active-learning label choice.
  3. CLASSIFICATION   — reframe the question as "is the reaction favourable?"
     (sign of ΔG) and a 3-class low/med/high ΔG bin, for virtual screening where the
     decision is categorical, not a precise number.

Everything is CPU-only and CV-based (no shipped model). Run under SLURM.

Usage
  python pipeline/explore_models.py                  # all three sections
  python pipeline/explore_models.py --skip uncertainty,classification
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

import delta_core as dc

warnings.filterwarnings("ignore")
SEED = 42


# ── shared CV on the Δ-correction → ΔG metrics ────────────────────────────────
def cv_regress(tbl, factory, folds=5, repeats=3, seed=SEED):
    """Repeated-K-fold OOF for any sklearn-compatible regressor of the Δ-correction.

    `factory()` returns a fresh unfitted estimator. Returns ΔG-level metrics
    (pred = dG_xtb + model(X)) and the OOF correction predictions."""
    X, y = tbl.X.to_numpy(), tbl.y
    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    oof_sum = np.zeros(len(X)); oof_cnt = np.zeros(len(X))
    for tr, te in rkf.split(X):
        m = factory(); m.fit(X[tr], y[tr])
        oof_sum[te] += m.predict(X[te]); oof_cnt[te] += 1
    oof = oof_sum / np.maximum(oof_cnt, 1)
    pred = tbl.dG_xtb + oof
    return (dict(MAE=float(mean_absolute_error(tbl.dG_dft, pred)),
                 RMSE=float(np.sqrt(np.mean((tbl.dG_dft - pred) ** 2))),
                 R2=float(r2_score(tbl.dG_dft, pred))), oof)


# ── 1. regression zoo ─────────────────────────────────────────────────────────
def regression_zoo(tbl):
    from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                                  HistGradientBoostingRegressor,
                                  RandomForestRegressor, StackingRegressor)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    def scaled(est):                       # distance/linear/kernel models need scaling
        return lambda: make_pipeline(StandardScaler(), est())

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1.0)
    zoo = {
        "xgb": lambda: XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                    random_state=SEED, n_jobs=-1),
        "hist_gbm": lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                          max_depth=None, random_state=SEED),
        "gbt": lambda: GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                                 learning_rate=0.05, subsample=0.8,
                                                 random_state=SEED),
        "random_forest": lambda: RandomForestRegressor(n_estimators=500, n_jobs=-1,
                                                        random_state=SEED),
        "extra_trees": lambda: ExtraTreesRegressor(n_estimators=500, n_jobs=-1,
                                                   random_state=SEED),
        "ridge": scaled(lambda: Ridge(alpha=10.0)),
        "elasticnet": scaled(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        "bayesian_ridge": scaled(lambda: BayesianRidge()),
        "svr_rbf": scaled(lambda: SVR(C=10.0, gamma="scale", epsilon=0.5)),
        "kernel_ridge": scaled(lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01)),
        "knn": scaled(lambda: KNeighborsRegressor(n_neighbors=10, weights="distance")),
        "mlp": scaled(lambda: MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                           max_iter=800, early_stopping=True,
                                           random_state=SEED)),
        "gpr": scaled(lambda: GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                                       normalize_y=True,
                                                       n_restarts_optimizer=0)),
    }
    try:
        from lightgbm import LGBMRegressor
        zoo["lightgbm"] = lambda: LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                                num_leaves=31, subsample=0.8,
                                                colsample_bytree=0.8, n_jobs=-1,
                                                random_state=SEED, verbose=-1)
    except ImportError:
        pass

    # Stacking the three best-bet base learners with a ridge meta-model.
    base = [("xgb", zoo["xgb"]()), ("rf", zoo["random_forest"]()),
            ("hist", zoo["hist_gbm"]())]
    zoo["stack"] = lambda: StackingRegressor(estimators=base, final_estimator=Ridge(),
                                             n_jobs=-1, passthrough=False)

    results = {}
    for name, factory in zoo.items():
        t0 = time.time()
        m, _ = cv_regress(tbl, factory)
        results[name] = {**m, "sec": round(time.time() - t0, 1)}
        print(f"  {name:16s} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R²={m['R2']:.3f}"
              f"  ({results[name]['sec']}s)")
    return results


# ── 2. uncertainty-aware regression ───────────────────────────────────────────
def gaussian_nll(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)))


def uncertainty_models(tbl, folds=5, repeats=2):
    """Probabilistic models that emit a predictive σ. Report point MAE, Gaussian
    NLL, and PICP — the fraction of true ΔG inside the nominal 80% interval
    (well-calibrated ≈ 0.80). σ here is the basis for active-learning selection."""
    X, y = tbl.X.to_numpy(), tbl.y
    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    out = {}

    def run(name, fit_predict):
        mu_s = np.zeros(len(X)); sg_s = np.zeros(len(X)); cnt = np.zeros(len(X))
        t0 = time.time()
        for tr, te in rkf.split(X):
            mu, sg = fit_predict(X[tr], y[tr], X[te])
            mu_s[te] += mu; sg_s[te] += sg; cnt[te] += 1
        mu = mu_s / np.maximum(cnt, 1); sg = sg_s / np.maximum(cnt, 1)
        pred = tbl.dG_xtb + mu                       # ΔG point estimate
        lo, hi = pred - 1.2816 * sg, pred + 1.2816 * sg   # 80% interval
        picp = float(np.mean((tbl.dG_dft >= lo) & (tbl.dG_dft <= hi)))
        out[name] = dict(MAE=float(mean_absolute_error(tbl.dG_dft, pred)),
                         NLL=gaussian_nll(y, mu, sg), PICP80=picp,
                         mean_sigma=float(np.mean(sg)), sec=round(time.time() - t0, 1))
        print(f"  {name:14s} MAE={out[name]['MAE']:.3f}  NLL={out[name]['NLL']:.3f}"
              f"  PICP80={picp:.2f} (target .80)  σ̄={out[name]['mean_sigma']:.2f}"
              f"  ({out[name]['sec']}s)")

    # NGBoost — native mean+variance.
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        def ngb(Xtr, ytr, Xte):
            m = NGBRegressor(Dist=Normal, n_estimators=500, learning_rate=0.01,
                             verbose=False, random_state=SEED)
            m.fit(Xtr, ytr)
            d = m.pred_dist(Xte)
            return d.loc, d.scale
        run("ngboost", ngb)
    except Exception as e:
        print(f"  ngboost skipped: {e}")

    # Gaussian Process — analytic posterior σ.
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        def gpr(Xtr, ytr, Xte):
            sc = StandardScaler().fit(Xtr)
            k = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1.0)
            g = GaussianProcessRegressor(kernel=k, alpha=1e-6, normalize_y=True,
                                         n_restarts_optimizer=0).fit(sc.transform(Xtr), ytr)
            mu, sd = g.predict(sc.transform(Xte), return_std=True)
            return mu, sd
        run("gpr", gpr)
    except Exception as e:
        print(f"  gpr skipped: {e}")

    # Quantile gradient boosting — interval from q=0.1/0.9, point from q=0.5.
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB

        def quant(Xtr, ytr, Xte):
            preds = {}
            for q in (0.1, 0.5, 0.9):
                m = HGB(loss="quantile", quantile=q, max_iter=400,
                        learning_rate=0.05, random_state=SEED).fit(Xtr, ytr)
                preds[q] = m.predict(Xte)
            mu = preds[0.5]
            sigma = (preds[0.9] - preds[0.1]) / (2 * 1.2816)   # 80% band → σ
            return mu, np.maximum(sigma, 1e-3)
        run("quantile_gbm", quant)
    except Exception as e:
        print(f"  quantile_gbm skipped: {e}")
    return out


# ── 3. classification reframing ───────────────────────────────────────────────
def classification(tbl, folds=5, repeats=3):
    """Reframe ΔG prediction as a categorical screening decision.

      • favourable : binary  dG_orca < 0   (reaction proceeds)
      • tercile    : 3-class low / med / high ΔG bins

    Features are the same 63 descriptors (NOT the Δ-target). Trees (xgb)."""
    from xgboost import XGBClassifier
    X = tbl.X.to_numpy()
    dG = tbl.dG_dft
    out = {}

    # binary favourability
    yb = (dG < 0).astype(int)
    rkf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    proba = np.zeros(len(X)); pred = np.zeros(len(X)); cnt = np.zeros(len(X))
    for tr, te in rkf.split(X, yb):
        c = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                          random_state=SEED, n_jobs=-1).fit(X[tr], yb[tr])
        proba[te] += c.predict_proba(X[te])[:, 1]; pred[te] += c.predict(X[te]); cnt[te] += 1
    proba /= np.maximum(cnt, 1); pred = (proba >= 0.5).astype(int)
    out["favourable_binary"] = dict(
        positive_rate=float(yb.mean()),
        accuracy=float(accuracy_score(yb, pred)), f1=float(f1_score(yb, pred)),
        roc_auc=float(roc_auc_score(yb, proba)))
    print(f"  favourable(ΔG<0): pos_rate={yb.mean():.2f} acc={out['favourable_binary']['accuracy']:.3f}"
          f" F1={out['favourable_binary']['f1']:.3f} AUC={out['favourable_binary']['roc_auc']:.3f}")

    # tercile multiclass
    q1, q2 = np.quantile(dG, [1/3, 2/3])
    ym = np.digitize(dG, [q1, q2])
    predm = np.zeros(len(X)); cnt = np.zeros(len(X))
    for tr, te in rkf.split(X, ym):
        c = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                          random_state=SEED, n_jobs=-1).fit(X[tr], ym[tr])
        predm[te] += c.predict(X[te]); cnt[te] += 1
    predm = np.round(predm / np.maximum(cnt, 1)).astype(int)
    out["tercile_3class"] = dict(
        cut_low=float(q1), cut_high=float(q2),
        accuracy=float(accuracy_score(ym, predm)),
        f1_macro=float(f1_score(ym, predm, average="macro")))
    print(f"  tercile(3-class): cuts=({q1:.1f},{q2:.1f}) acc={out['tercile_3class']['accuracy']:.3f}"
          f" F1_macro={out['tercile_3class']['f1_macro']:.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", default="", help="comma list: zoo,uncertainty,classification")
    ap.add_argument("--outdir", default=str(dc.REPO_ROOT / "runs"))
    args = ap.parse_args()
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    tbl = dc.load_training_table()
    print(f"\nTable: n={len(tbl.df)}  features={len(tbl.feats)}  target={tbl.target}")
    base_mae = float(mean_absolute_error(tbl.dG_dft, tbl.dG_xtb))
    print(f"xTB baseline MAE = {base_mae:.2f} kcal/mol\n")

    res = {"n": len(tbl.df), "n_features": len(tbl.feats), "xtb_baseline_mae": base_mae}
    if "zoo" not in skip:
        print("── 1. REGRESSION ZOO ──────────────────────────────────────────")
        res["regression"] = regression_zoo(tbl)
    if "uncertainty" not in skip:
        print("\n── 2. UNCERTAINTY-AWARE REGRESSION ────────────────────────────")
        res["uncertainty"] = uncertainty_models(tbl)
    if "classification" not in skip:
        print("\n── 3. CLASSIFICATION REFRAMING ────────────────────────────────")
        res["classification"] = classification(tbl)

    out = Path(args.outdir); (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "data" / "model_exploration.json", "w"), indent=2)
    print(f"\nSaved {out / 'data' / 'model_exploration.json'}")

    # Bar chart of regression-zoo MAE.
    if "regression" in res:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        items = sorted(res["regression"].items(), key=lambda kv: kv[1]["MAE"])
        names = [k for k, _ in items]; maes = [v["MAE"] for _, v in items]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names, maes, color="C0")
        ax.axvline(base_mae, color="r", ls="--", label=f"xTB baseline {base_mae:.1f}")
        for i, m in enumerate(maes):
            ax.text(m, i, f" {m:.2f}", va="center", fontsize=8)
        ax.set_xlabel("CV MAE vs r2SCAN-3c ΔG (kcal/mol)"); ax.invert_yaxis()
        ax.set_title(f"Regression model zoo (n={len(tbl.df)})"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "figs" / "model_zoo.png", dpi=140)
        print(f"Saved {out / 'figs' / 'model_zoo.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
