#!/usr/bin/env python3
"""
G — Train ONE hybrid D-MPNN (Chemprop) Δ-learning model and log it to MLflow.

Same target + comparison harness as train_delta.py, but the model is a graph
message-passing net that also ingests the 62 descriptors + dG_xtb as extra
features (see gnn_core). Logs CV metrics vs the xTB baseline, a parity figure,
the CV predictions and the fitted model to the **same** MLflow experiment
(`benzoin_delta_dG`, tag stage="G-gnn") so it sits side-by-side with the trees.

Usage
  python pipeline/gnn/train_gnn.py
  python pipeline/gnn/train_gnn.py --params '{"depth":5,"ensemble":3}' --repeats 2
  python pipeline/gnn/train_gnn.py --target dG_orca_shermo_kcal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np

# delta_core + gnn_core live alongside this file's parent (pipeline/) and here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import delta_core as dc  # noqa: E402
import gnn_core as gc  # noqa: E402


def make_parity(tbl, pred_dft, m_base, m_delta, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    dG = tbl.dG_dft
    lo, hi = float(min(dG.min(), pred_dft.min())), float(max(dG.max(), pred_dft.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.scatter(dG, tbl.dG_xtb, s=14, alpha=0.4, label=f"xTB (MAE {m_base['MAE']:.2f})")
    ax.scatter(dG, pred_dft, s=14, alpha=0.7, label=f"D-MPNN (MAE {m_delta['MAE']:.2f})")
    ax.set_xlabel(f"{tbl.target} (kcal/mol)")
    ax.set_ylabel("predicted ΔG (kcal/mol)")
    ax.legend(); ax.set_title("Benzoin ΔG — hybrid D-MPNN CV parity")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desc-glob", default=dc.DEFAULT_DESC_GLOB)
    ap.add_argument("--label-glob", default=dc.DEFAULT_LABEL_GLOB)
    ap.add_argument("--target", default=dc.DEFAULT_TARGET,
                    choices=["dG_orca_kcal", "dG_orca_shermo_kcal"])
    ap.add_argument("--params", default="{}", help="JSON dict of gnn_core param overrides")
    ap.add_argument("--parquet", default=dc.DEFAULT_FEATURIZE_PARQUET,
                    help="training table; point at an alternate label version "
                         "(e.g. data/featurize_funnel.parquet) for apples-to-apples")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--outdir", default=str(dc.REPO_ROOT / "runs"))
    args = ap.parse_args()

    params = {**gc.DEFAULT_PARAMS, **json.loads(args.params)}
    out = Path(args.outdir)
    (out / "models_gnn").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    print("Loading descriptors + labels...")
    tbl = dc.load_training_table(args.desc_glob, args.label_glob, args.target,
                                 parquet=args.parquet)
    print(f"Joined rows: {len(tbl.df):,}   x_d features: {len(tbl.feats)}   target: {tbl.target}")
    if len(tbl.df) < args.folds * 3:
        print("Not enough joined rows to train yet — are the subset jobs finished?")
        return 1

    print(f"\nRunning {args.repeats}×{args.folds}-fold CV (hybrid D-MPNN)...")
    delta, base, pred_dft = gc.cv_evaluate_gnn(
        tbl, params, args.folds, args.repeats, args.seed)
    print(f"\n{args.repeats}×{args.folds}-fold CV vs {tbl.target}:")
    print(f"  xTB baseline   MAE={base['MAE']:6.2f}  RMSE={base['RMSE']:6.2f}  R²={base['R2']:6.3f}")
    print(f"  D-MPNN (hybrid)MAE={delta['MAE']:6.2f}  RMSE={delta['RMSE']:6.2f}  R²={delta['R2']:6.3f}")
    print(f"  MAE improvement: {base['MAE'] - delta['MAE']:+.2f} kcal/mol")

    parity = out / "figs" / "parity_gnn.png"
    make_parity(tbl, pred_dft, base, delta, parity)

    cv_csv = out / "data" / "cv_predictions_gnn.csv"
    dfo = tbl.df[["index", "SMILES", dc.XTB_DG, tbl.target]].copy()
    dfo["dG_pred"] = pred_dft
    dfo.to_csv(cv_csv, index=False)

    # Final fit on all data + save ensemble and scalers.
    print("\nFitting final model on all data...")
    models, scalers = gc.fit_full(tbl, params, args.seed)
    import torch
    torch.save([m.state_dict() for m in models], out / "models_gnn" / "mpnn_state.pt")
    joblib.dump({"x_scaler": scalers["x_scaler"], "y_scaler": scalers["y_scaler"]},
                out / "models_gnn" / "scalers.joblib")
    meta = {
        "model": "dmpnn_hybrid", "params": params, "target": tbl.target,
        "n_samples": int(len(tbl.df)), "n_xd_features": len(tbl.feats),
        "feature_list": tbl.feats, "folds": args.folds, "repeats": args.repeats,
        "cv_xtb_baseline": base, "cv_delta_learning": delta,
        "feature_medians": tbl.medians,
    }
    (out / "models_gnn" / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ── MLflow (same experiment as the trees) ────────────────────────────────
    dc.setup_mlflow()
    tag = tbl.target.replace("dG_", "").replace("_kcal", "")
    run_name = args.run_name or f"dmpnn-{tag}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"target": tbl.target, "model_family": "dmpnn",
                         "stage": "G-gnn", "n_features": len(tbl.feats)})
        mlflow.log_params({"model": "dmpnn_hybrid", "folds": args.folds,
                           "repeats": args.repeats, "seed": args.seed,
                           "n_samples": len(tbl.df), "n_xd_features": len(tbl.feats),
                           **{f"hp_{k}": v for k, v in params.items()}})
        mlflow.log_metrics({
            "cv_mae": delta["MAE"], "cv_rmse": delta["RMSE"], "cv_r2": delta["R2"],
            "base_mae": base["MAE"], "base_rmse": base["RMSE"], "base_r2": base["R2"],
            "mae_improvement": base["MAE"] - delta["MAE"],
        })
        mlflow.log_artifact(str(parity), "figs")
        mlflow.log_artifact(str(cv_csv), "data")
        mlflow.log_artifact(str(out / "models_gnn" / "metadata.json"), "model")
        mlflow.log_artifact(str(out / "models_gnn" / "mpnn_state.pt"), "model")
        mlflow.log_artifact(str(out / "models_gnn" / "scalers.joblib"), "model")
        print(f"\nLogged MLflow run '{run_name}' to {dc.TRACKING_URI} (exp '{dc.EXPERIMENT}')")

    print(f"Saved model + metadata to {out/'models_gnn'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
