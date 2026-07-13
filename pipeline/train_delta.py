#!/usr/bin/env python3
"""
D — Train ONE benzoin Δ-learning model and log it to MLflow.

Predicts the DFT-level correction on top of the cheap xTB ΔG:
    target  y  = dG_dft − dG_xtb_kcal
    predict    = dG_xtb_kcal + model(X)  ≈ ORCA PBE0-D4 ΔG

Logs params, repeated-K-fold CV metrics (vs the pure-xTB baseline), parity +
SHAP-importance figures, the CV predictions and the fitted model to MLflow so
runs are comparable in the UI. For a hyperparameter search use sweep_delta.py.

Usage
  python ml/train_delta.py                       # xgb defaults
  python ml/train_delta.py --model rf --repeats 5
  python ml/train_delta.py --target dG_orca_shermo_kcal
  # then:  mlflow ui --backend-store-uri sqlite:///scratch-shared/schen3/benzoin-dg/mlflow.db
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np

import delta_core as dc


def make_parity(tbl, pred_dft, m_base, m_delta, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    dG = tbl.dG_dft
    lo, hi = float(min(dG.min(), pred_dft.min())), float(max(dG.max(), pred_dft.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.scatter(dG, tbl.dG_xtb, s=14, alpha=0.4, label=f"xTB (MAE {m_base['MAE']:.2f})")
    ax.scatter(dG, pred_dft, s=14, alpha=0.7, label=f"Δ-learn (MAE {m_delta['MAE']:.2f})")
    ax.set_xlabel(f"{tbl.target} (kcal/mol)")
    ax.set_ylabel("predicted ΔG (kcal/mol)")
    ax.legend(); ax.set_title("Benzoin ΔG — CV parity")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def make_shap(model, X, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    expl = shap.Explainer(model, X)
    sv = expl(X)
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desc-glob", default=dc.DEFAULT_DESC_GLOB)
    ap.add_argument("--label-glob", default=dc.DEFAULT_LABEL_GLOB)
    ap.add_argument("--target", default=dc.DEFAULT_TARGET,
                    choices=["dG_orca_kcal", "dG_orca_shermo_kcal"])
    ap.add_argument("--model", default="xgb", choices=["xgb", "rf", "gbt"])
    ap.add_argument("--params", default="{}", help="JSON dict of model overrides")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--outdir", default=str(dc.REPO_ROOT / "runs"))
    ap.add_argument("--parquet", default=dc.DEFAULT_FEATURIZE_PARQUET,
                    help="training table (descriptors + baseline + target on one geometry)")
    ap.add_argument("--baseline", default="gfn2",
                    help="semiempirical baseline in dG_xtb_kcal: 'gfn2' or 'gxtb_cosmo_dmso' "
                         "(recorded in metadata; inference reads it to match)")
    args = ap.parse_args()

    params = json.loads(args.params)
    out = Path(args.outdir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    print("Loading descriptors + labels...")
    tbl = dc.load_training_table(args.desc_glob, args.label_glob, args.target,
                                 parquet=args.parquet)
    print(f"Joined rows: {len(tbl.df):,}   features: {len(tbl.feats)}   target: {tbl.target}")
    if len(tbl.df) < args.folds * 3:
        print("Not enough joined rows to train yet — are the subset jobs finished?")
        return 1

    delta, base, pred_dft = dc.cv_evaluate(
        tbl, args.model, params, args.folds, args.repeats, args.seed)
    print(f"\n{args.repeats}×{args.folds}-fold CV vs {tbl.target}:")
    print(f"  xTB baseline   MAE={base['MAE']:6.2f}  RMSE={base['RMSE']:6.2f}  R²={base['R2']:6.3f}")
    print(f"  Δ-learning     MAE={delta['MAE']:6.2f}  RMSE={delta['RMSE']:6.2f}  R²={delta['R2']:6.3f}")
    print(f"  MAE improvement: {base['MAE'] - delta['MAE']:+.2f} kcal/mol")

    # Final fit on all data.
    final = dc.build_model(args.model, params, args.seed)
    final.fit(tbl.X, tbl.y)

    parity = out / "figs" / "parity.png"
    make_parity(tbl, pred_dft, base, delta, parity)
    importance = out / "figs" / "importance.png"
    try:
        make_shap(final, tbl.X, importance)
    except Exception as e:
        print(f"(SHAP skipped: {e})"); importance = None

    cv_csv = out / "data" / "cv_predictions.csv"
    dfo = tbl.df[["index", "SMILES", dc.XTB_DG, tbl.target]].copy()
    dfo["dG_pred"] = pred_dft
    dfo.to_csv(cv_csv, index=False)
    joblib.dump(final, out / "models" / "delta_model.joblib")
    (out / "models" / "feature_list.json").write_text(json.dumps(tbl.feats, indent=2))
    (out / "models" / "metadata.json").write_text(json.dumps({
        "model": args.model, "params": params, "target": tbl.target,
        "baseline": args.baseline,
        "n_samples": int(len(tbl.df)), "n_features": len(tbl.feats),
        "folds": args.folds, "repeats": args.repeats,
        "cv_xtb_baseline": base, "cv_delta_learning": delta,
        "feature_medians": tbl.medians,
    }, indent=2))

    # ── MLflow ──────────────────────────────────────────────────────────────
    dc.setup_mlflow()
    run_name = args.run_name or f"{args.model}-{tbl.target.replace('dG_','').replace('_kcal','')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"target": tbl.target, "model_family": args.model,
                         "stage": "D", "n_features": len(tbl.feats)})
        mlflow.log_params({"model": args.model, "folds": args.folds,
                           "repeats": args.repeats, "seed": args.seed,
                           "n_samples": len(tbl.df), "n_features": len(tbl.feats),
                           **{f"hp_{k}": v for k, v in params.items()}})
        mlflow.log_metrics({
            "cv_mae": delta["MAE"], "cv_rmse": delta["RMSE"], "cv_r2": delta["R2"],
            "base_mae": base["MAE"], "base_rmse": base["RMSE"], "base_r2": base["R2"],
            "mae_improvement": base["MAE"] - delta["MAE"],
        })
        mlflow.log_artifact(str(parity), "figs")
        if importance:
            mlflow.log_artifact(str(importance), "figs")
        mlflow.log_artifact(str(cv_csv), "data")
        mlflow.log_artifact(str(out / "models" / "feature_list.json"), "model")
        mlflow.log_artifact(str(out / "models" / "metadata.json"), "model")
        if args.model == "xgb":
            mlflow.xgboost.log_model(final, name="model")
        else:
            mlflow.sklearn.log_model(final, name="model")
        print(f"\nLogged MLflow run '{run_name}' to {dc.TRACKING_URI} (exp '{dc.EXPERIMENT}')")

    print(f"Saved model + metadata to {out/'models'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
