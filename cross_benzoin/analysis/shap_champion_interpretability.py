#!/usr/bin/env python
"""
First proper SHAP interpretability pass for the CURRENT cross-benzoin production
champion (single-XGB, scaffold_disjoint_v1, 80/10/10 split, 260 features) -- the only
prior SHAP work for cross was a stale round3-scale importance CSV
(cross_round3/train_3rounds_mordred_v1/data/shap_importance_full.csv, 564 raw features,
pre-scaffold-fix, pre-mordred-slim) and shap_slim_cross_mordred.py, which used SHAP only
as a feature-SELECTION tool (pruning), not as an interpretability deliverable. Mirrors
homo's viz_gxtb_20260625/ SHAP suite (importance ranking + dependence plots), scoped down
to what's actually informative at this stage (full waterfall-per-molecule and interaction
heatmaps deferred -- not asked for, importance ranking + block-level grouping + top-feature
dependence plots directly answer "which descriptors matter and are they interpretable").

Also directly informs the donor/acceptor-asymmetry question (see memory
cross-five-diagnostics-20260717.md's flagged-open finding): block-level SHAP importance
(donor_* vs acceptor_* vs product-related vs global) shows whether the role-aware feature
architecture is actually using both roles' information, not just one.

Usage
  python cross_benzoin/analysis/shap_champion_interpretability.py \
      --model-dir data/cross_benzoin/cross_round7/scaffold_disjoint_v1 \
      --table data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_split_labeled.parquet \
      --outdir data/cross_benzoin/cross_round7/shap_champion_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "pipeline"))

# single-hue sequential ramp (magnitude job), light->dark, matches this repo's existing
# figure style (no rainbow, no dual-axis)
BAR_COLOR = "#2b6cb0"
BLOCK_COLORS = {"donor": "#2b6cb0", "acceptor": "#c05621", "product-related": "#2f855a",
                "global": "#805ad5"}


def block_of(feat: str) -> str:
    if feat.startswith("donor_"):
        return "donor"
    if feat.startswith("acceptor_"):
        return "acceptor"
    if feat.startswith("product_"):
        return "product-related"
    # un-prefixed xtb_/mulliken_/wbo_/fukui_/etc columns are computed directly on the
    # PRODUCT molecule (its own reacting-atom descriptors) -- see assemble_cross_training_table.py
    return "product-related" if any(feat.startswith(p) for p in
        ("xtb_", "mulliken_", "wbo_", "fukui_", "dual_", "vbur_", "hb_", "dih", "pa",
         "SASA", "P_int", "bde")) else "global"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-sample", type=int, default=5000,
                     help="subsample train rows for SHAP (exact TreeExplainer is fast but "
                          "keep this bounded for a quick, reproducible pass)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model_dir / "models" / "champion_scaffold_disjoint.joblib")
    feats = json.loads((args.model_dir / "models" / "feature_list.json").read_text())

    df = pd.read_parquet(args.table)
    train_df = df[df["new_scaffold_split"] == "train"].reset_index(drop=True)
    if len(train_df) > args.n_sample:
        train_df = train_df.sample(n=args.n_sample, random_state=args.seed).reset_index(drop=True)
    print(f"SHAP on {len(train_df)} training rows, {len(feats)} features")

    medians = train_df[feats].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    X = train_df[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)

    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs, "block": [block_of(f) for f in feats]})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(args.outdir / "shap_importance_full.csv", index=False)
    print(f"wrote {args.outdir/'shap_importance_full.csv'}")

    block_imp = imp.groupby("block")["mean_abs_shap"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
    block_imp.to_csv(args.outdir / "shap_block_importance.csv")
    print("\n=== block-level SHAP importance (sum of mean|SHAP| within block) ===")
    print(block_imp.to_string())

    # --- top-20 bar chart (sequential/magnitude job -> single hue, ordered) ---
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color=BAR_COLOR, height=0.65)
    ax.set_xlabel("mean |SHAP value| (kcal/mol contribution to prediction)")
    ax.set_title("Cross-benzoin champion (single-XGB, scaffold-disjoint 80/10/10):\ntop 20 features by SHAP importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.outdir / "shap_top20_importance.png", dpi=150)
    plt.close(fig)

    # --- block-level bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4))
    bi = block_imp.reset_index()
    colors = [BLOCK_COLORS.get(b, "#718096") for b in bi["block"]]
    ax.bar(bi["block"], bi["sum"], color=colors, width=0.6)
    ax.set_ylabel("summed mean |SHAP| within block")
    ax.set_title("SHAP importance by feature block\n(donor vs acceptor vs product-related vs global)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.outdir / "shap_block_importance.png", dpi=150)
    plt.close(fig)

    # --- dependence plots for top 5 individual features ---
    for i, feat in enumerate(imp["feature"].head(5)):
        fig, ax = plt.subplots(figsize=(5, 4))
        fi = feats.index(feat)
        ax.scatter(X[feat], sv[:, fi], s=6, alpha=0.35, color=BAR_COLOR, edgecolors="none")
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value (kcal/mol)")
        ax.set_title(f"SHAP dependence: {feat}\n(rank {i+1} by importance)")
        ax.axhline(0, color="#a0aec0", lw=1, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.outdir / f"shap_dependence_rank{i+1:02d}_{feat}.png", dpi=150)
        plt.close(fig)

    print(f"\nwrote plots -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
