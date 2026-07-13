#!/usr/bin/env python3
"""
Standalone "full journey" figure: the whole benzoin ΔG accuracy arc, 2026-06-11→29.

Two eras, deliberately NOT joined by a line because the metric/target differ:
  Era 1 — aldehyde-side Δ-learning, repeated 5×3-fold CV MAE vs DFT ΔG.
          187 → funnel_v3 (2.23) → g-xTB base swap (2.00).
  Era 2 — cross-benzoin PRODUCT-side g-xTB→DFT correction, 70/20/10 held-out
          TEST MAE. Production benchmark 2026-06-26: MLP+XGB ensemble wins
          (all 1.61 / aromatic 1.42); pure-graph GNN ~2.6 loses.

All numbers are read from benzoin-dg/mlflow.db (Era 1, key cv_mae) and
benzoin-dg/mlflow_benchmark.db (Era 2, key test_mae / mae_aromatic), 2026-06-29.
The vertical divider + the y-label note make explicit that Era-1 CV MAE and Era-2
held-out test MAE are NOT directly comparable (different target + validation).
One figure per file; fresh filename so prior figures are never overwritten.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figs" / "cvmae_full_journey_20260629.png"

# (x, label, value, kind)
# Era 1 kinds: main / gnn / scope / base(g-xTB swap)   Era 2 kinds: ens / ensA / gnn2
PTS = [
    (0,  "187 xgb\nKMeans-200",       3.070, "main"),
    (1,  "472 xgb\nMaxMin",            3.235, "main"),
    (2,  "D-MPNN\nn=1038",            3.486, "gnn"),
    (3,  "D-MPNN\nn=1427",            3.401, "gnn"),
    (4,  "1500 xgb",                  3.215, "main"),
    (5,  "1500-clean\nxgb",          3.159, "main"),
    (6,  "aromatic\n(scope,dropped)", 2.681, "scope"),
    (7,  "funnel v1\nxgb",           2.291, "main"),
    (8,  "funnel_v3\nn=1634",        2.228, "main"),
    (9,  "funnel_v3\nn=3061",        2.228, "main"),
    (10, "g-xTB base\nn=1644",       2.000, "base"),
    # ---- divider between 10 and 11 ----
    (12, "ensemble\nall",            1.614, "ens"),
    (13, "ensemble\naromatic",       1.419, "ensA"),
    (14, "pure-GNN\n(graph only)",   2.583, "gnn2"),
]
DIVIDER = 11.0

STYLE = {
    "main":  dict(color="#1f4e79", marker="o", s=70,  label="tree Δ-model (full scope)"),
    "gnn":   dict(color="#c0392b", marker="X", s=95,  label="hybrid D-MPNN (failed)"),
    "scope": dict(color="#888888", marker="^", s=80,  label="aromatic-only (dropped)"),
    "base":  dict(color="#2e86c1", marker="D", s=85,  label="g-xTB base swap"),
    "ens":   dict(color="#1e7d34", marker="*", s=240, label="MLP+XGB ensemble (champion)"),
    "ensA":  dict(color="#27ae60", marker="*", s=190, label="ensemble — aromatic"),
    "gnn2":  dict(color="#c0392b", marker="X", s=95,  label="pure-graph GNN (lost again)"),
}


def main() -> int:
    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Era-1 noise-floor band
    ax.axhspan(3.06, 3.49, xmax=(DIVIDER + 0.5) / 15.5, color="#d9d9d9", alpha=0.4, zorder=0)
    ax.text(2.0, 3.52, "label-noise plateau — adding data doesn't help",
            fontsize=8.5, style="italic", color="#555")

    # Era-1 main trajectory line (only same-metric, full-scope/base points)
    line_kinds = {"main", "base"}
    lx = [p[0] for p in PTS if p[3] in line_kinds]
    ly = [p[2] for p in PTS if p[3] in line_kinds]
    ax.plot(lx, ly, "-", color="#1f4e79", lw=1.4, alpha=0.55, zorder=2)

    # vertical divider between eras
    ax.axvline(DIVIDER, color="#444", ls=":", lw=1.3, zorder=1)

    seen = set()
    for x, _lbl, y, kind in PTS:
        st = dict(STYLE[kind]); lab = st.pop("label")
        ax.scatter([x], [y], **st, label=lab if lab not in seen else None,
                   edgecolors="white", linewidths=0.8, zorder=3)
        seen.add(lab)
        off = -16 if kind in ("scope",) else 10
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, off), ha="center", fontsize=8.5,
                    fontweight="bold", color=st["color"])

    # breakthrough arrow (Era 1)
    ax.annotate("conformer-search fix\n(funnel v1→v3)",
                xy=(7, 2.30), xytext=(4.3, 2.46), fontsize=9, color="#1e7d34",
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1e7d34", lw=1.5))

    # era band titles
    ax.text(5.0, 3.66, "Era 1 — aldehyde-side Δ-learning   (repeated 5×3-fold CV MAE)",
            fontsize=10, ha="center", color="#1f4e79", fontweight="bold")
    ax.text(13.0, 3.66, "Era 2 — cross-benzoin product side\n(g-xTB→DFT, 70/20/10 TEST MAE)",
            fontsize=9.5, ha="center", color="#1e7d34", fontweight="bold")

    ax.set_xticks([p[0] for p in PTS])
    ax.set_xticklabels([p[1] for p in PTS], fontsize=7.6)
    ax.set_ylabel("MAE vs DFT  (kcal/mol)  — lower is better\n"
                  "[Era 1 = CV MAE,  Era 2 = held-out test MAE — NOT directly comparable]",
                  fontsize=9.5)
    ax.set_xlabel("milestone (chronological order →)", fontsize=10)
    ax.set_title("Benzoin ΔG accuracy — full journey (2026-06-11 → 06-29):  "
                 "cleaning > architecture, then better baseline + product-side features",
                 fontsize=11.5)
    ax.set_ylim(1.2, 3.75)
    ax.set_xlim(-0.6, 14.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8.2, framealpha=0.95, ncol=2)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
