#!/usr/bin/env python
"""Feature-importance (XGBoost GAIN) for the key 72-feature g-xTB->DFT Δ-models.

Reads the SAVED production bundle (ENSEMBLE72: MLP+XGB_d8+XGB_d10) and the Optuna-tuned
XGB; plots gain importance per tree model, bars colored by descriptor family
(product-QM / reactant-QM / RDKit-global). One standalone PNG per model (repo convention),
new filenames only. CPU, project env. Outputs -> data/cross_benzoin/homo_v6/viz_gxtb_20260625/.
"""
import joblib, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

R = Path("/scratch-shared/schen3/benzoin-dg")
OUT = R / "data/cross_benzoin/homo_v6/viz_gxtb_20260625"
TOPN = 25

# --- descriptor family of each feature name (for color) ---
def family(f):
    if f.startswith("ald_"): return "reactant QM"
    if f.startswith("g_"):   return "RDKit global"
    return "product QM"
FCOLOR = {"product QM": "#2171b5", "reactant QM": "#cb181d", "RDKit global": "#238b45"}

def gain_importance(model, feats):
    """XGBoost gain per feature, mapped fXX -> name; 0 for unused features."""
    score = model.get_booster().get_score(importance_type="gain")  # {'f0':g,...}
    g = np.array([score.get(f"f{i}", 0.0) for i in range(len(feats))])
    return g / g.sum() * 100.0  # percent

def plot(model, feats, title, fname):
    imp = gain_importance(model, feats)
    order = np.argsort(imp)[::-1][:TOPN][::-1]   # top-N, ascending for barh
    names = [feats[i] for i in order]
    vals = imp[order]
    cols = [FCOLOR[family(n)] for n in names]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(range(len(names)), vals, color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("XGBoost gain importance (% of total)")
    ax.set_title(title, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FCOLOR.values()]
    ax.legend(handles, FCOLOR.keys(), loc="lower right", fontsize=9, title="descriptor family")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)
    print("wrote", fname, "| top5:", ", ".join(f"{names[-1-k]}={vals[-1-k]:.1f}" for k in range(5)), flush=True)

# 1-2) ensemble members
b = joblib.load(R / "pipeline/models/gxtb_dft_correction_ENSEMBLE72_20260626.joblib")
feats = b["features"]
mem = dict(b["members"])
plot(mem["XGB_d8"],  feats, "Feature importance — XGB_d8 (ENSEMBLE72 member)\ng-xTB->DFT Δ-learning, 72 feats", "30_featimp_xgb_d8.png")
plot(mem["XGB_d10"], feats, "Feature importance — XGB_d10 (ENSEMBLE72 member)\ng-xTB->DFT Δ-learning, 72 feats", "31_featimp_xgb_d10.png")

# 3) Optuna-tuned best single XGB
o = joblib.load(R / "pipeline/models/xgb_optuna_tuned_20260626.joblib")
plot(o["model"], o["features"], "Feature importance — XGB Optuna-tuned (best single)\ng-xTB->DFT Δ-learning, 72 feats", "32_featimp_xgb_optuna.png")

print("done")
