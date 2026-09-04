#!/usr/bin/env python
"""
Verify the repackaged CrossBenzoinBlendPredictor (round1-10 champion/ensemble +
attentive-pooling GNN) reproduces the training-time MAE, then bootstrap the
blend-vs-ensemble-only delta for significance. Mirrors verify_and_bootstrap_8rounds.py's
method, applied to the round1-10 retrain.
"""
import sys
import json

sys.path.insert(0, "cross_benzoin")
import pandas as pd
import numpy as np
from predict_cross_champion import CrossBenzoinBlendPredictor

df = pd.read_parquet(
    "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet")
test = df[df["new_scaffold_split"] == "test"].reset_index(drop=True)
print("test n =", len(test))

pred = CrossBenzoinBlendPredictor.load(
    "data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_v1",
    gnn_dir="data/cross_benzoin/cross_round10/gnn_attentive_10rounds_v1",
)
dG_pred = pred.predict(test)
actual = test["dG_orca_kcal"].to_numpy()
err_blend = np.abs(dG_pred - actual)
mae_blend = err_blend.mean()
print(f"packaged predictor MAE on 10rounds test (n={len(test)}): {mae_blend:.4f} "
      f"(reported best_blend_mae=2.074)")

ens_delta = pred.ensemble.predict(test)
base = test["dG_gxtb_kcal"].to_numpy()
ens_pred = base + ens_delta
err_ens = np.abs(ens_pred - actual)
mae_ens = err_ens.mean()
print(f"ensemble-only MAE: {mae_ens:.4f}  (reported 2.163)")

rng = np.random.default_rng(20260721)
B = 20000
n = len(test)
deltas = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n, n)
    deltas[b] = err_ens[idx].mean() - err_blend[idx].mean()
ci = np.percentile(deltas, [5, 95])
p_better = (deltas > 0).mean()
point_delta = mae_ens - mae_blend
result = {
    "n": n,
    "point_mae_ensemble": float(mae_ens),
    "point_mae_blend": float(mae_blend),
    "point_delta": float(point_delta),
    "bootstrap_90ci_delta": [float(ci[0]), float(ci[1])],
    "p_blend_better_than_ensemble": float(p_better),
    "blend_w_gnn": float(pred.blend_w_gnn),
}
print(json.dumps(result, indent=2))
outp = "data/cross_benzoin/cross_round10/gnn_attentive_10rounds_v1/bootstrap_significance.json"
open(outp, "w").write(json.dumps(result, indent=2))
print("wrote", outp)
