#!/usr/bin/env python3
"""
D-sweep — Optuna hyperparameter / model-family search for the Δ-learning model,
every trial logged as a nested MLflow run for side-by-side comparison.

Objective = repeated-K-fold CV MAE of the Δ-learning ΔG vs the DFT target
(minimised). Repeated K-fold is what makes the objective stable enough at n≈200
that Optuna optimises real signal rather than CV-split noise.

After the search it refits the best config on all data, logs it as the parent
run's model, and writes ml/models/{delta_model,feature_list,metadata}.

Usage
  python ml/sweep_delta.py --trials 60
  python ml/sweep_delta.py --trials 80 --model xgb           # pin a family
  python ml/sweep_delta.py --trials 60 --target dG_orca_shermo_kcal
  # then:  mlflow ui --backend-store-uri sqlite:///scratch-shared/schen3/benzoin-dg/mlflow.db
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import optuna

import delta_core as dc


def suggest(trial: optuna.Trial, family: str) -> tuple[str, dict]:
    if family == "all":
        family = trial.suggest_categorical("model", ["xgb", "rf", "gbt"])
    if family == "xgb":
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        )
    elif family == "rf":
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
        )
    else:  # gbt
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
        )
    return family, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desc-glob", default=dc.DEFAULT_DESC_GLOB)
    ap.add_argument("--label-glob", default=dc.DEFAULT_LABEL_GLOB)
    ap.add_argument("--parquet", default=dc.DEFAULT_FEATURIZE_PARQUET,
                    help="training table; point at an alternate label version "
                         "(e.g. data/featurize_funnel.parquet) to compare cleanly")
    ap.add_argument("--target", default=dc.DEFAULT_TARGET,
                    choices=["dG_orca_kcal", "dG_orca_shermo_kcal"])
    ap.add_argument("--model", default="all", choices=["all", "xgb", "rf", "gbt"])
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(dc.REPO_ROOT / "runs"))
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    print("Loading descriptors + labels...")
    tbl = dc.load_training_table(args.desc_glob, args.label_glob, args.target,
                                 parquet=args.parquet)
    print(f"Joined rows: {len(tbl.df):,}   features: {len(tbl.feats)}   target: {tbl.target}")
    if len(tbl.df) < args.folds * 3:
        print("Not enough joined rows to train yet — are the subset jobs finished?")
        return 1
    _, base, _ = dc.cv_evaluate(tbl, "xgb", {}, args.folds, args.repeats, args.seed)
    print(f"xTB baseline MAE = {base['MAE']:.2f} kcal/mol  (target to beat)")

    dc.setup_mlflow()
    parent = mlflow.start_run(run_name=f"sweep-{args.model}-{tbl.target.replace('dG_','').replace('_kcal','')}")
    mlflow.set_tags({"stage": "D-sweep", "target": tbl.target, "search_family": args.model})

    def objective(trial: optuna.Trial) -> float:
        family, params = suggest(trial, args.model)
        delta, _, _ = dc.cv_evaluate(tbl, family, params, args.folds, args.repeats, args.seed)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.set_tags({"model_family": family, "target": tbl.target})
            mlflow.log_params({"model": family, **{f"hp_{k}": v for k, v in params.items()}})
            mlflow.log_metrics({"cv_mae": delta["MAE"], "cv_rmse": delta["RMSE"],
                                "cv_r2": delta["R2"],
                                "mae_improvement": base["MAE"] - delta["MAE"]})
        trial.set_user_attr("family", family)
        trial.set_user_attr("params", params)
        return delta["MAE"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    family = best.user_attrs["family"]
    params = best.user_attrs["params"]
    delta, _, pred_dft = dc.cv_evaluate(tbl, family, params, args.folds, args.repeats, args.seed)
    print(f"\nBest: {family}  CV MAE={delta['MAE']:.2f}  RMSE={delta['RMSE']:.2f}  R²={delta['R2']:.3f}")
    print(f"  improvement vs xTB: {base['MAE'] - delta['MAE']:+.2f} kcal/mol")
    print(f"  params: {json.dumps(params)}")

    final = dc.build_model(family, params, args.seed)
    final.fit(tbl.X, tbl.y)
    joblib.dump(final, out / "models" / "delta_model.joblib")
    (out / "models" / "feature_list.json").write_text(json.dumps(tbl.feats, indent=2))
    (out / "models" / "metadata.json").write_text(json.dumps({
        "model": family, "params": params, "target": tbl.target,
        "n_samples": int(len(tbl.df)), "n_features": len(tbl.feats),
        "folds": args.folds, "repeats": args.repeats, "trials": args.trials,
        "cv_xtb_baseline": base, "cv_delta_learning": delta,
        "feature_medians": tbl.medians,
    }, indent=2))

    # Log best to the parent run.
    mlflow.set_tags({"best_model_family": family})
    mlflow.log_params({"best_model": family, **{f"best_hp_{k}": v for k, v in params.items()}})
    mlflow.log_metrics({"best_cv_mae": delta["MAE"], "best_cv_rmse": delta["RMSE"],
                        "best_cv_r2": delta["R2"], "base_mae": base["MAE"],
                        "best_mae_improvement": base["MAE"] - delta["MAE"]})
    mlflow.log_artifact(str(out / "models" / "feature_list.json"), "model")
    mlflow.log_artifact(str(out / "models" / "metadata.json"), "model")
    try:
        import optuna.visualization.matplotlib as ov
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for fn, name in [(ov.plot_optimization_history, "opt_history"),
                         (ov.plot_param_importances, "param_importances")]:
            try:
                fn(study); plt.tight_layout()
                p = out / "figs" / f"{name}.png"
                plt.savefig(p, dpi=130); plt.close()
                mlflow.log_artifact(str(p), "figs")
            except Exception:
                plt.close()
    except Exception as e:
        print(f"(optuna plots skipped: {e})")
    if family == "xgb":
        mlflow.xgboost.log_model(final, name="model")
    else:
        mlflow.sklearn.log_model(final, name="model")
    mlflow.end_run()

    print(f"\nLogged sweep to {dc.TRACKING_URI} (exp '{dc.EXPERIMENT}')")
    print(f"Saved best model to {out/'models'/'delta_model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
