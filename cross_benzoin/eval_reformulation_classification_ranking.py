#!/usr/bin/env python
"""Goal-3 reformulation probe: does casting dG prediction as classification /
ranking sidestep the ~2.9 kcal single-conformer DFT label-noise ceiling found
by confnoise_cross (RUN_LOG 09-04 ~14:38)?

Post-hoc only (no retraining): load the frozen r1-10 champion blend's
continuous predictions on its own scaffold-disjoint holdout (n=448) against
the true dG_orca_kcal label, then score two reformulations:
  1. Binary favorable/unfavorable at threshold T in {0, median(train)} --
     accuracy / ROC-AUC / F1 from simply thresholding the *existing*
     regressor's output (no new model).
  2. Ranking quality -- Spearman rho (already implied by R² but reported
     directly) + top-decile precision (of the true-best 10% of pairs by
     dG_orca, what fraction does the model's top-decile-by-prediction catch).

If accuracy/AUC/top-decile precision are high despite MAE 2.215 on a ~2.9 kcal
noise floor, that's the case for shipping "rank/screen" instead of "predict
the number" -- a metric that is robust to the label noise by construction.

Usage:
  /home/schen3/venv/nequip/bin/python cross_benzoin/eval_reformulation_classification_ranking.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import spearmanr

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
sys.path.insert(0, str(REPO / "cross_benzoin"))
from predict_cross_champion import CrossBenzoinBlendPredictor  # noqa: E402

TABLE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
MODEL_DIR = REPO / "data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_v1"
GNN_DIR = REPO / "data/cross_benzoin/cross_round10/gnn_attentive_10rounds_v1"


def top_decile_precision(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.1) -> float:
    n = len(y_true)
    k = max(1, int(round(n * frac)))
    true_top = set(np.argsort(y_true)[:k])          # most negative / most favorable dG
    pred_top = set(np.argsort(y_pred)[:k])
    return len(true_top & pred_top) / k


def main() -> int:
    df = pd.read_parquet(TABLE)
    test = df[df["new_scaffold_split"] == "test"].reset_index(drop=True)
    print(f"holdout n={len(test)}")

    pred = CrossBenzoinBlendPredictor.load(str(MODEL_DIR), gnn_dir=str(GNN_DIR))
    y_pred = pred.predict(test)
    y_true = test["dG_orca_kcal"].to_numpy()
    gxtb = test["dG_gxtb_kcal"].to_numpy() if "dG_gxtb_kcal" in test.columns else None

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rho, _ = spearmanr(y_true, y_pred)
    print(f"regression sanity: blend MAE={mae:.3f}  Spearman rho={rho:.3f}")

    out = {"n": len(test), "regression_mae": mae, "spearman_rho": float(rho)}

    for thr_name, thr in [("T=0", 0.0), ("T=train_median", float(df.loc[df["new_scaffold_split"] == "train", "dG_orca_kcal"].median()))]:
        yt = (y_true < thr).astype(int)
        if yt.sum() in (0, len(yt)):
            print(f"  [{thr_name}] degenerate (all one class), skip")
            continue
        yp_score = -y_pred  # more favorable (more negative) -> higher score
        yp_hat = (y_pred < thr).astype(int)
        auc = roc_auc_score(yt, yp_score)
        acc = accuracy_score(yt, yp_hat)
        f1 = f1_score(yt, yp_hat)
        # same, but thresholding the raw g-xTB baseline instead of the ML model
        base_auc = base_acc = base_f1 = None
        if gxtb is not None:
            base_hat = (gxtb < thr).astype(int)
            base_auc = roc_auc_score(yt, -gxtb)
            base_acc = accuracy_score(yt, base_hat)
            base_f1 = f1_score(yt, base_hat)
        print(f"  [{thr_name}={thr:.2f}] positive_rate={yt.mean():.2f}  "
              f"ML: AUC={auc:.3f} acc={acc:.3f} F1={f1:.3f}"
              + (f"  |  g-xTB baseline: AUC={base_auc:.3f} acc={base_acc:.3f} F1={base_f1:.3f}" if gxtb is not None else ""))
        out[thr_name] = {"threshold": thr, "positive_rate": float(yt.mean()),
                          "ml_auc": float(auc), "ml_acc": float(acc), "ml_f1": float(f1),
                          "gxtb_auc": float(base_auc) if base_auc is not None else None,
                          "gxtb_acc": float(base_acc) if base_acc is not None else None,
                          "gxtb_f1": float(base_f1) if base_f1 is not None else None}

    for frac in (0.1, 0.2):
        p = top_decile_precision(y_true, y_pred, frac)
        p_gxtb = top_decile_precision(y_true, gxtb, frac) if gxtb is not None else None
        print(f"  top-{int(frac*100)}% precision: ML={p:.3f}"
              + (f"  g-xTB baseline={p_gxtb:.3f}" if p_gxtb is not None else ""))
        out[f"top{int(frac*100)}pct_precision_ml"] = p
        out[f"top{int(frac*100)}pct_precision_gxtb"] = p_gxtb

    outp = REPO / "data/cross_benzoin/reformulation_classification_ranking_eval.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
