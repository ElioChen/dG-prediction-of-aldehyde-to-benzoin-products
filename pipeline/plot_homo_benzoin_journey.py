#!/usr/bin/env python3
"""
homo-benzoin figure (g-xTB ONLY): the g-xTB→DFT ΔG correction benchmark for the
homo-coupling benzoin model. Companion to plot_screen_v6_models.py.

Single task, single metric: 70/20/10 held-out TEST MAE vs r2SCAN-3c DFT ΔG
(kcal/mol), all on the g-xTB baseline (NO GFN2-era / aldehyde-side points here).

Story:
  • raw g-xTB vs DFT = 4.26 kcal (GFN2 = 15.7, off-scale) — the error to correct;
  • a learned correction with growing product+reactant feature sets f34→f56→f72,
    across model families Ridge/MLP/XGB/Ensemble, drives TEST MAE down to 1.61
    (aromatic 1.42) — MLP+XGB ensemble on f72 is champion;
  • reference architectures (3D DimeNet++, GINE-hybrid, dual-QM GNN, pure-SMILES
    Tier-1) don't beat the tabular ensemble (dual-QM GNN only ties, using the 56
    QM features fed in) — info > architecture.

Numbers from benzoin-dg/mlflow_benchmark.db, experiment
gxtb_dft_full_benchmark_20260626 (keys test_mae / mae_aromatic), read 2026-06-29.
One figure per file; fresh filename so prior figures are never overwritten.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figs" / "homo_benzoin_gxtb_correction_20260629.png"

RAW_GXTB = 4.258          # raw g-xTB vs DFT, no correction (all scope)
RAW_GFN2 = 15.701         # reference, off-scale

# feature-set x positions (product-only 34 → +reactant 56 → +global 72)
FX = {"f34\nprod": 0, "f56\nprod+react": 1, "f72\n+global": 2}

# model family -> {feature set: test_mae(all)}, color, marker
FAMILIES = {
    "Ridge":    (dict(zip(FX, [2.797, 2.143, 2.105])), "#7f7f7f", "o"),
    "MLP":      (dict(zip(FX, [2.328, 1.727, 1.686])), "#2e86c1", "s"),
    "XGB":      (dict(zip(FX, [2.243, 1.717, 1.679])), "#e67e22", "^"),
    "Ensemble": (dict(zip(FX, [2.242, 1.648, 1.614])), "#1e7d34", "D"),
}
ENS_AROM = dict(zip(FX, [1.945, 1.449, 1.419]))     # ensemble aromatic-only

# reference architectures (x, label, test_mae)
REFS = [
    (4.0, "dual-QM\nGNN",      1.616, "#16a085"),
    (4.8, "3D\nDimeNet++",     2.040, "#8e44ad"),
    (5.6, "GINE\nhybrid",      2.130, "#c0392b"),
    (6.4, "Tier-1\npure-SMILES", 2.750, "#95a5a6"),
]
DIVIDER = 3.2


def main() -> int:
    fig, ax = plt.subplots(figsize=(12.5, 6.6))

    # raw g-xTB starting line
    ax.axhline(RAW_GXTB, color="#444", ls="--", lw=1.3, zorder=1)
    ax.text(0.0, RAW_GXTB + 0.06, f"raw g-xTB vs DFT = {RAW_GXTB:.2f}  "
            f"(GFN2 = {RAW_GFN2:.1f}, off-scale)  →  correction target",
            fontsize=9, color="#444", va="bottom")

    # family lines across feature sets
    xs = list(FX.values())
    for fam, (vals, col, mk) in FAMILIES.items():
        ys = [vals[k] for k in FX]
        ax.plot(xs, ys, "-", color=col, marker=mk, ms=9, lw=1.7,
                mec="white", label=fam, zorder=3)
        for k in FX:
            ax.annotate(f"{vals[k]:.2f}", (FX[k], vals[k]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.8,
                        fontweight="bold", color=col)
    # ensemble aromatic (faint, dashed)
    ax.plot(xs, [ENS_AROM[k] for k in FX], ":", color="#27ae60", marker="*",
            ms=12, lw=1.3, alpha=0.85, label="Ensemble — aromatic", zorder=3)
    f72_key = list(FX)[-1]
    ax.annotate(f"{ENS_AROM[f72_key]:.2f}", (2, ENS_AROM[f72_key]),
                textcoords="offset points", xytext=(10, -4), fontsize=8,
                fontweight="bold", color="#27ae60")

    # champion halo
    ax.scatter([2], [1.614], s=420, facecolors="none", edgecolors="#1e7d34",
               linewidths=2.0, zorder=2)
    ax.annotate("champion\nMLP+XGB ensemble", (2, 1.614), textcoords="offset points",
                xytext=(18, 14), fontsize=8.5, color="#1e7d34", fontweight="bold")

    # divider + reference architectures
    ax.axvline(DIVIDER, color="#444", ls=":", lw=1.3, zorder=1)
    for x, lab, v, col in REFS:
        ax.scatter([x], [v], color=col, marker="P", s=120, edgecolors="white",
                   linewidths=0.8, zorder=3)
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.3, fontweight="bold", color=col)

    ax.text(1.0, 3.0, "learned g-xTB→DFT correction\n(feature set × model family)",
            fontsize=10, ha="center", color="#1f4e79", fontweight="bold")
    ax.text(5.2, 3.45, "reference architectures\n(don't beat the ensemble)",
            fontsize=10, ha="center", color="#8e44ad", fontweight="bold")

    ax.set_xticks(list(FX.values()) + [r[0] for r in REFS])
    ax.set_xticklabels(list(FX.keys()) + [r[1] for r in REFS], fontsize=8)
    ax.set_ylabel("TEST MAE vs DFT ΔG  (kcal/mol, 70/20/10 hold-out)  — lower is better",
                  fontsize=9.5)
    ax.set_xlabel("feature set (more info →)        |        reference models", fontsize=10)
    ax.set_title("homo-benzoin: g-xTB→DFT ΔG correction benchmark (2026-06-26)\n"
                 "raw 4.26 → ensemble 1.61 (aromatic 1.42); info > architecture",
                 fontsize=11.5)
    ax.set_ylim(1.3, 4.5)
    ax.set_xlim(-0.5, 6.9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.40, 0.90), fontsize=8.4,
              framealpha=0.95, ncol=3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
