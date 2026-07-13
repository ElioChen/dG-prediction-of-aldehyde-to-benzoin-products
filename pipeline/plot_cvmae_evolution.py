#!/usr/bin/env python3
"""
Standalone figure: the cv_mae evolution of the benzoin Δ-learning model, from the
first 187-molecule run to the shipped funnel_v3 production model.

Tells the project's core story in one plot:
  • adding data (187→472→1500) does NOT lower cv_mae — a label-noise plateau ~3.2;
  • the hybrid D-MPNN side-branch sits ABOVE the plateau (architecture didn't help);
  • fixing the CONFORMER SEARCH (funnel v1 → v3) breaks through to ~2.23.

All cv_mae values are the best (lowest) repeated-5×3-fold CV MAE logged for each
stage in benzoin-dg/mlflow.db (experiment family benzoin_delta_dG*), read 2026-06-29.
One figure per file (no composite panels); written to a fresh filename so prior
figures are never overwritten.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figs" / "cvmae_evolution_nodate_20260629.png"

# (order, date, label, cv_mae, kind)  kind ∈ {main, gnn, scope}
#   main  = full-scope tree (xgb) Δ-model on that label/subset version
#   gnn   = hybrid D-MPNN side-branch (failed vs trees)
#   scope = aromatic-only restricted scope (later dropped — not apples-to-apples)
PTS = [
    (0, "6/11", "187 xgb\nKMeans-200",     3.070, "main"),
    (1, "6/12", "472 xgb\nMaxMin",          3.235, "main"),
    (2, "6/12", "D-MPNN\nn=1038",           3.486, "gnn"),
    (3, "6/12", "D-MPNN\nn=1427",           3.401, "gnn"),
    (4, "6/13", "1500 xgb",                 3.215, "main"),
    (5, "6/13", "1500-clean\nxgb",          3.159, "main"),
    (6, "6/13", "aromatic xgb\n(scope, dropped)", 2.681, "scope"),
    (7, "6/13", "funnel v1\nxgb",           2.291, "main"),
    (8, "6/15", "funnel_v3\nprod n=1634",   2.228, "main"),
    (9, "6/15+", "funnel_v3\nn=3061",       2.228, "main"),
]

STYLE = {
    "main":  dict(color="#1f4e79", marker="o", s=70, zorder=3, label="tree Δ-model (full scope)"),
    "gnn":   dict(color="#c0392b", marker="X", s=95, zorder=3, label="hybrid D-MPNN (failed)"),
    "scope": dict(color="#888888", marker="^", s=80, zorder=3, label="aromatic-only (dropped)"),
}


def main() -> int:
    fig, ax = plt.subplots(figsize=(10, 6))

    # noise-floor band over the "more data didn't help" plateau
    ax.axhspan(3.06, 3.49, color="#d9d9d9", alpha=0.45, zorder=0)
    ax.text(2.0, 3.50, "label-noise plateau — adding data doesn't help",
            fontsize=9, style="italic", color="#555555", va="bottom")

    # main-trajectory connecting line (full-scope tree runs, in date order)
    main_x = [p[0] for p in PTS if p[4] == "main"]
    main_y = [p[3] for p in PTS if p[4] == "main"]
    ax.plot(main_x, main_y, "-", color="#1f4e79", lw=1.4, alpha=0.6, zorder=2)

    seen = set()
    for x, _date, _lbl, y, kind in PTS:
        st = dict(STYLE[kind])
        lab = st.pop("label")
        ax.scatter([x], [y], **st, label=lab if lab not in seen else None,
                   edgecolors="white", linewidths=0.8)
        seen.add(lab)
        dy = 0.06 if kind != "gnn" else 0.05
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 9 if kind != "scope" else -16), ha="center",
                    fontsize=8.5, fontweight="bold", color=st["color"])

    # breakthrough arrow: plateau -> funnel
    ax.annotate("conformer-search fix\n(funnel v1→v3)",
                xy=(7, 2.30), xytext=(4.3, 2.45), fontsize=9, color="#1e7d34",
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1e7d34", lw=1.6))

    ax.set_xticks([p[0] for p in PTS])
    ax.set_xticklabels([p[2] for p in PTS], fontsize=8.2)
    ax.set_ylabel("CV MAE vs DFT ΔG  (kcal/mol)  — lower is better", fontsize=10)
    ax.set_xlabel("milestone (chronological order →)", fontsize=10)
    ax.set_title("Benzoin Δ-learning: cv_mae evolution\n"
                 "187 → funnel_v3 (cleaning > architecture ≈ more data)", fontsize=12)
    ax.set_ylim(2.0, 3.7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
