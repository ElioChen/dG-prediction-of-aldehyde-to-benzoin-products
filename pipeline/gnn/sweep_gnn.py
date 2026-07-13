#!/usr/bin/env python3
"""
G-sweep — Optuna search over hybrid D-MPNN (Chemprop) hyperparameters, each trial
a nested MLflow run, objective = repeated-K-fold CV MAE of the Δ-learning ΔG.

Same MLflow experiment as the tree sweep (`benzoin_delta_dG`) so the GNN trials
are directly comparable to xgb/rf/gbt. Meant to run on a gpu_a100 node
(submit_gnn.sh); each CV is fast, so the A100 budget goes to breadth (trials ×
ensemble) rather than a single long train.

Usage
  python pipeline/gnn/sweep_gnn.py --trials 40
  python pipeline/gnn/sweep_gnn.py --trials 30 --repeats 2 --ensemble-max 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import optuna

# delta_core + gnn_core live alongside this file's parent (pipeline/) and here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import delta_core as dc  # noqa: E402
import gnn_core as gc  # noqa: E402


def suggest(trial: optuna.Trial, ensemble_max: int) -> dict:
    return dict(
        depth=trial.suggest_int("depth", 2, 6),
        message_hidden=trial.suggest_categorical("message_hidden", [150, 300, 600]),
        ffn_hidden=trial.suggest_categorical("ffn_hidden", [150, 300, 600]),
        ffn_layers=trial.suggest_int("ffn_layers", 1, 3),
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        max_epochs=trial.suggest_categorical("max_epochs", [80, 120, 200]),
        ensemble=trial.suggest_int("ensemble", 1, ensemble_max),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desc-glob", default=dc.DEFAULT_DESC_GLOB)
    ap.add_argument("--label-glob", default=dc.DEFAULT_LABEL_GLOB)
    ap.add_argument("--target", default=dc.DEFAULT_TARGET,
                    choices=["dG_orca_kcal", "dG_orca_shermo_kcal"])
    ap.add_argument("--parquet", default=dc.DEFAULT_FEATURIZE_PARQUET,
                    help="training table; alternate label version for comparison")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--ensemble-max", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(dc.REPO_ROOT / "runs"))
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "models_gnn").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    print("Loading descriptors + labels...")
    tbl = dc.load_training_table(args.desc_glob, args.label_glob, args.target,
                                 parquet=args.parquet)
    print(f"Joined rows: {len(tbl.df):,}   x_d features: {len(tbl.feats)}   target: {tbl.target}")
    if len(tbl.df) < args.folds * 3:
        print("Not enough joined rows to train yet — are the subset jobs finished?")
        return 1
    _, base, _ = gc.cv_evaluate_gnn(tbl, {"max_epochs": 1, "ensemble": 1},
                                    args.folds, 1, args.seed, verbose=False)
    print(f"xTB baseline MAE = {base['MAE']:.2f} kcal/mol  (target to beat)")

    dc.setup_mlflow()
    parent = mlflow.start_run(run_name=f"sweep-dmpnn-{tbl.target.replace('dG_','').replace('_kcal','')}")
    mlflow.set_tags({"stage": "G-sweep", "target": tbl.target, "search_family": "dmpnn"})

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, args.ensemble_max)
        delta, _, _ = gc.cv_evaluate_gnn(tbl, params, args.folds, args.repeats,
                                         args.seed, verbose=False)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.set_tags({"model_family": "dmpnn", "target": tbl.target})
            mlflow.log_params({"model": "dmpnn", **{f"hp_{k}": v for k, v in params.items()}})
            mlflow.log_metrics({"cv_mae": delta["MAE"], "cv_rmse": delta["RMSE"],
                                "cv_r2": delta["R2"],
                                "mae_improvement": base["MAE"] - delta["MAE"]})
        trial.set_user_attr("params", params)
        print(f"trial {trial.number}: MAE={delta['MAE']:.3f}  {json.dumps(params)}", flush=True)
        return delta["MAE"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_trial
    params = best.user_attrs["params"]
    delta, _, pred_dft = gc.cv_evaluate_gnn(tbl, params, args.folds, args.repeats,
                                            args.seed, verbose=False)
    print(f"\nBest D-MPNN  CV MAE={delta['MAE']:.2f}  RMSE={delta['RMSE']:.2f}  R²={delta['R2']:.3f}")
    print(f"  improvement vs xTB: {base['MAE'] - delta['MAE']:+.2f} kcal/mol")
    print(f"  params: {json.dumps(params)}")

    # Refit on all data + save.
    models, scalers = gc.fit_full(tbl, params, args.seed)
    import torch
    torch.save([m.state_dict() for m in models], out / "models_gnn" / "mpnn_state.pt")
    joblib.dump({"x_scaler": scalers["x_scaler"], "y_scaler": scalers["y_scaler"]},
                out / "models_gnn" / "scalers.joblib")
    (out / "models_gnn" / "metadata.json").write_text(json.dumps({
        "model": "dmpnn_hybrid", "params": params, "target": tbl.target,
        "n_samples": int(len(tbl.df)), "n_xd_features": len(tbl.feats),
        "feature_list": tbl.feats, "folds": args.folds, "repeats": args.repeats,
        "trials": args.trials, "cv_xtb_baseline": base, "cv_delta_learning": delta,
        "feature_medians": tbl.medians,
    }, indent=2))

    mlflow.set_tags({"best_model_family": "dmpnn"})
    mlflow.log_params({"best_model": "dmpnn", **{f"best_hp_{k}": v for k, v in params.items()}})
    mlflow.log_metrics({"best_cv_mae": delta["MAE"], "best_cv_rmse": delta["RMSE"],
                        "best_cv_r2": delta["R2"], "base_mae": base["MAE"],
                        "best_mae_improvement": base["MAE"] - delta["MAE"]})
    mlflow.log_artifact(str(out / "models_gnn" / "metadata.json"), "model")
    mlflow.end_run()

    print(f"\nLogged sweep to {dc.TRACKING_URI} (exp '{dc.EXPERIMENT}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
