#!/usr/bin/env python3
"""
screen-v6 figure: MAE of every surrogate model trained to predict the xTB ΔG
(`dG_xtb_kcal`) of the full v6 screen library — the "ΔG simulatability ceiling" study.

This is NOT Δ-learning: the target is the cheap xTB ΔG itself, modelled from 24-31
xTB/morféus (+ADCH/QTAIM) scalar descriptors. The story: linear sits well above a
~2.0 kcal aromatic-MAE ceiling; MLP / GBT / GINE-GNN all pile up at that ceiling;
the 2D-graph GNN does NOT beat GBT — a REPRESENTATION bottleneck that justified
moving to 3D/QM features + DFT labels (the homo-benzoin track).

Numbers read from pipeline/docs/REPORT_screen_v6_models_20260629.md (2026-06-29):
  Group A — in-distribution model scan (aromatic / all-scope MAE)
  Group B — honest Bemis–Murcko scaffold split + GINE-GNN vs GBT (aromatic MAE)
One figure per file; fresh filename so prior figures are never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figs" / "screen_v6_model_mae_20260629.png"
OUT_NOSCAF = REPO / "figs" / "screen_v6_model_mae_noscaffold_20260629.png"
OUT_GNN = REPO / "figs" / "screen_v6_gnn_vs_gbt_20260629.png"

# GINE-GNN vs GBT on the SAME Bemis–Murcko scaffold split (aromatic, n≈146,741)
# (test MAE, R², RMSE) — read from REPORT_screen_v6_models_20260629.md §3.
GNN_CMP = [
    ("GBT\nscaffold-random",  2.13, 0.680, None,  "#1e7d34"),   # easier representative split
    ("GINE-GNN\n(2D graph)",  2.18, 0.611, 3.10,  "#c0392b"),
    ("GBT\n(31 desc)",        2.14, 0.631, 3.02,  "#1e7d34"),
    ("GBT\nscaffold-RARE",    2.58, 0.542, None,  "#8e44ad"),   # hardest split
]


def plot_gnn() -> Path:
    """Focused GNN view: 2D-graph GINE-GNN vs GBT on the same scaffold split."""
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.axhspan(2.0, 2.18, color="#fde6c4", alpha=0.7, zorder=0)
    ax.text(0.0, 2.015, "non-linear ceiling ≈ 2.0 (aromatic)",
            fontsize=9, style="italic", color="#9c5b00", va="bottom", ha="left")
    xs = list(range(len(GNN_CMP)))
    for x, (lab, mae, r2, rmse, col) in zip(xs, GNN_CMP):
        mk = "X" if "GNN" in lab else ("v" if "RARE" in lab else "s")
        ax.scatter([x], [mae], color=col, marker=mk, s=160, edgecolors="white",
                   linewidths=1.0, zorder=3)
        tag = f"MAE {mae:.2f}\nR² {r2:.3f}" + (f"\nRMSE {rmse:.2f}" if rmse else "")
        ax.annotate(tag, (x, mae), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8.6, fontweight="bold", color=col)
    # highlight the head-to-head pair (GINE-GNN vs GBT-31, same split)
    ax.annotate("", xy=(2, 2.14), xytext=(1, 2.18),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.3))
    ax.text(1.5, 2.45, "same scaffold split:\nGNN 2.18  >  GBT 2.14\n(2D graph loses)",
            ha="center", fontsize=9.5, color="#333", fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.9))
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in GNN_CMP], fontsize=8.4)
    ax.set_ylabel("scaffold-split TEST MAE of predicted xTB ΔG  (kcal/mol)", fontsize=9.5)
    ax.set_xlabel("model (Bemis–Murcko scaffold split, aromatic n≈146,741)", fontsize=9.5)
    ax.set_title("screen-v6: 2D-graph GINE-GNN does NOT beat GBT\n"
                 "GIN(E) is WL-max expressive, yet scalar-descriptor GBT still wins → representation, not model, is the ceiling",
                 fontsize=10.8)
    ax.set_ylim(1.9, 2.75)
    ax.set_xlim(-0.6, 3.6)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    OUT_GNN.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GNN, dpi=140)
    return OUT_GNN

# Group A: in-distribution (random) model scan — (x, label, aromatic_MAE, all_MAE)
GROUP_A = [
    (0, "linear OLS\n24 feat",      2.35, 2.63),
    (1, "MLP\n(128,64)",            2.02, 2.21),
    (2, "MLP tuned\n(256,128)",     2.05, 2.31),
    (3, "GBT\n24 feat",             2.01, 2.27),
    (4, "GBT +ADCH\n/QTAIM 31f",    2.06, 2.21),
]
# Group B: honest scaffold split + GNN — (x, label, aromatic_MAE, kind)
GROUP_B = [
    (6, "GBT\nscaffold-rand",  2.13, "gbt"),
    (7, "GINE-GNN\nscaffold",  2.18, "gnn"),
    (8, "GBT 31f\nscaffold",   2.14, "gbt"),
    (9, "GBT scaffold\n-RARE (hardest)", 2.58, "hard"),
]
DIVIDER = 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-scaffold", action="store_true",
                    help="Drop Group B (scaffold split + GNN); plot only the "
                         "in-distribution model-family scan.")
    ap.add_argument("--gnn", action="store_true",
                    help="Focused GINE-GNN vs GBT (same scaffold split) figure.")
    args = ap.parse_args()
    if args.gnn:
        out = plot_gnn()
        print(f"Wrote {out}")
        return 0
    scaffold = not args.no_scaffold

    fig, ax = plt.subplots(figsize=(12 if scaffold else 8.5, 6.3))

    # representation-ceiling band (aromatic)
    ceil_x = 2.0 if scaffold else 2.0
    ax.axhspan(2.0, 2.18, color="#fde6c4", alpha=0.7, zorder=0)
    ax.text(ceil_x, 2.185, "non-linear ceiling ≈ 2.0 (aromatic) — representation bottleneck",
            fontsize=9, style="italic", color="#9c5b00", va="bottom", ha="center")

    # Group A — aromatic (primary, line) + all-scope (faint hollow)
    ax_x = [p[0] for p in GROUP_A]
    ax.plot(ax_x, [p[2] for p in GROUP_A], "-o", color="#1f4e79", lw=1.6, ms=9,
            mec="white", label="aromatic MAE (random split)", zorder=3)
    ax.scatter(ax_x, [p[3] for p in GROUP_A], facecolors="none",
               edgecolors="#7f7f7f", s=70, label="all-scope MAE", zorder=3)
    for x, _l, ar, al in GROUP_A:
        ax.annotate(f"{ar:.2f}", (x, ar), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, fontweight="bold", color="#1f4e79")
        ax.annotate(f"{al:.2f}", (x, al), textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=7.5, color="#7f7f7f")

    if scaffold:
        # divider
        ax.axvline(DIVIDER, color="#444", ls=":", lw=1.3, zorder=1)

        # Group B — honest scaffold split
        cmap = {"gbt": ("#1e7d34", "s", "GBT (scaffold split)"),
                "gnn": ("#c0392b", "X", "GINE-GNN (2D graph — loses)"),
                "hard": ("#8e44ad", "v", "hardest rare-scaffold")}
        seen = set()
        for x, _l, ar, kind in GROUP_B:
            c, mk, lab = cmap[kind]
            ax.scatter([x], [ar], color=c, marker=mk, s=110, edgecolors="white",
                       linewidths=0.8, zorder=3, label=lab if lab not in seen else None)
            seen.add(lab)
            ax.annotate(f"{ar:.2f}", (x, ar), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8.5, fontweight="bold", color=c)

        # group band titles
        ax.text(2.0, 2.66, "Group A — in-distribution model scan",
                fontsize=10, ha="center", color="#1f4e79", fontweight="bold")
        ax.text(7.5, 2.66, "Group B — honest scaffold split + GNN",
                fontsize=10, ha="center", color="#1e7d34", fontweight="bold")

    xt = [p[0] for p in GROUP_A] + ([p[0] for p in GROUP_B] if scaffold else [])
    xl = [p[1] for p in GROUP_A] + ([p[1] for p in GROUP_B] if scaffold else [])
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7.8)
    ax.set_ylabel("MAE of predicted xTB ΔG  (kcal/mol)  — lower is better", fontsize=10)
    ax.set_xlabel("surrogate model (development order →)", fontsize=10)
    if scaffold:
        ax.set_title("screen-v6: xTB-ΔG surrogate models — a representation-ceiling study\n"
                     "linear → MLP/GBT/GNN all plateau ~2.0 (aromatic); 2D-graph GNN doesn't beat GBT",
                     fontsize=11.5)
    else:
        ax.set_title("screen-v6: xTB-ΔG surrogate models (in-distribution model scan)\n"
                     "linear 2.35 → MLP/GBT plateau ~2.0 (aromatic); ADCH/QTAIM no real gain",
                     fontsize=11.5)
    ax.set_ylim(1.85, 2.72)
    ax.set_xlim(-0.6, 9.8 if scaffold else 4.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8.4, framealpha=0.95, ncol=2)

    fig.tight_layout()
    out = OUT if scaffold else OUT_NOSCAF
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
