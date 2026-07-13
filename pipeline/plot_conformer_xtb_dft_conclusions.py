#!/usr/bin/env python3
"""
Visualize the three conclusions of the conformer / xTB-DFT synthesis (2026-06-22,
REPORT_conformer_xtb_dft_synthesis_20260622.md). ONE standalone figure per file
(no multi-panel composites), dated filenames (history preserved).

  C1a  conformer/geometry choice swings the DFT reaction ΔE  (~8 kcal MAD)
  C1b  g-xTB is NOT a better geometry engine (GFN2 geom gives lower DFT E ~98%)
  C2   g-xTB energy ≈ DFT (MAE 3.3, r .97), an order of magnitude better than GFN2
  C3   DFT convergence is the blocker; g-xTB geometry unblocks DFT-SP
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/gpfs/scratch1/shared/schen3")
FIGS = ROOT / "benzoin-dg/figs"
STAMP = "20260622"
GX = pd.read_csv(ROOT / "gxtb_test/gxtbgeom_dftsp_1pct_all_20260621.csv")
MG = pd.read_csv(ROOT / "benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c/analysis/"
                 "dft_sp_merged_20260618_002414.csv")

def _mae(a, b): return float(np.mean(np.abs(a - b)))
def _r(a, b):   return float(np.corrcoef(a, b)[0, 1])


# ── C1a: geometry/conformer choice swings the DFT reaction ΔE ────────────────
def fig_c1a():
    d = GX.dropna(subset=["dE_dft_gxtbgeom_kcal", "dE_dft_xtbgeom_kcal"])
    diff = (d.dE_dft_gxtbgeom_kcal - d.dE_dft_xtbgeom_kcal).to_numpy()
    mad = float(np.median(np.abs(diff))); frac5 = float(np.mean(np.abs(diff) > 5))
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(diff.min(), diff.max(), 120)   # full range, no clipping
    ax.hist(diff, bins=bins, color="#3b6ea5", edgecolor="white", lw=0.2)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("DFT reaction ΔE:  (g-xTB geometry) − (GFN2 geometry)   [kcal/mol]")
    ax.set_ylabel("molecules")
    ax.set_title("Effect of geometry choice on the DFT reaction ΔE\n"
                 "(same r2SCAN-3c, same gas phase; only the geometry differs)", fontsize=11)
    ax.text(0.02, 0.97, f"n = {len(diff)}\nMAD = {mad:.1f} kcal/mol\n"
            f"{frac5*100:.0f}% differ by >5 kcal/mol\n"
            f"range {diff.min():.0f} … {diff.max():.0f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout(); out = FIGS / f"geometry_swings_dft_dE_{STAMP}.png"
    fig.savefig(out, dpi=150); plt.close(fig); return out


# ── C1b: g-xTB is NOT a better geometry engine ───────────────────────────────
def fig_c1b():
    H = 627.509
    d = GX.dropna(subset=["E_ald_dft_gxtbgeom_Eh", "E_ald_dft_xtbgeom_Eh",
                          "E_bz_dft_gxtbgeom_Eh", "E_bz_dft_xtbgeom_Eh"]).copy()
    # DFT single-point energy PENALTY of using g-xTB geom instead of GFN2 geom (kcal);
    # >0 means g-xTB geometry is variationally worse (higher DFT energy).
    pa = (d.E_ald_dft_gxtbgeom_Eh - d.E_ald_dft_xtbgeom_Eh) * H
    pb = (d.E_bz_dft_gxtbgeom_Eh - d.E_bz_dft_xtbgeom_Eh) * H
    both = float(np.mean((pa > 0) & (pb > 0)))
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.axhline(0, color="0.5", lw=1); ax.axvline(0, color="0.5", lw=1)
    ax.scatter(pa, pb, s=12, alpha=0.5, color="#1f6fb2", edgecolor="none")
    ax.set_xlabel("aldehyde:  DFT energy penalty of g-xTB geom  [kcal/mol]")
    ax.set_ylabel("benzoin product:  DFT energy penalty of g-xTB geom  [kcal/mol]")
    ax.set_title("Per-molecule DFT energy penalty of g-xTB vs GFN2 geometry\n"
                 "(penalty > 0  ⇒  GFN2/funnel geometry is variationally lower)",
                 fontsize=11)
    ax.text(0.97, 0.05,
            f"n = {len(d)}\n{both*100:.0f}% in the upper-right quadrant\n"
            f"(g-xTB geom worse for BOTH species)\n"
            f"median penalty: ald {float(pa.median()):.0f}, product {float(pb.median()):.0f}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout(); out = FIGS / f"gxtb_geometry_penalty_scatter_{STAMP}.png"
    fig.savefig(out, dpi=150); plt.close(fig); return out


# ── C2: g-xTB energy ≈ DFT, an order of magnitude better than GFN2 ────────────
def fig_c2():
    g = GX.dropna(subset=["dG_gxtb_kcal", "dG_dft_gxtbgeom_kcal"])
    gx_x, gx_y = g.dG_gxtb_kcal.to_numpy(), g.dG_dft_gxtbgeom_kcal.to_numpy()
    m = MG.dropna(subset=["dG_xtb_kcal", "dG_orca_kcal"])
    f2_x, f2_y = m.dG_xtb_kcal.to_numpy(), m.dG_orca_kcal.to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    lim = (-55, 55); ax.plot(lim, lim, "k-", lw=1, zorder=1)
    ax.scatter(f2_x, f2_y, s=9, alpha=0.35, color="#888888", zorder=2,
               label=f"GFN2  (MAE {_mae(f2_x,f2_y):.1f}, r {_r(f2_x,f2_y):.2f})")
    ax.scatter(gx_x, gx_y, s=9, alpha=0.55, color="#1f6fb2", zorder=3,
               label=f"g-xTB (MAE {_mae(gx_x,gx_y):.1f}, r {_r(gx_x,gx_y):.2f})")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("semiempirical reaction ΔG  [kcal/mol]")
    ax.set_ylabel("DFT r2SCAN-3c reaction ΔG  [kcal/mol]")
    ax.set_title("Semiempirical vs DFT reaction ΔG, each at its own fixed geometry\n"
                 "(1% set; g-xTB: g-xTB geom/gas · GFN2: GFN2 geom/DMSO)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    fig.tight_layout(); out = FIGS / f"semiempirical_vs_dft_dG_{STAMP}.png"
    fig.savefig(out, dpi=150); plt.close(fig); return out


# ── C3: DFT convergence is the blocker; g-xTB geometry unblocks DFT-SP ────────
def fig_c3():
    n_sp = int((GX.E_bz_dft_gxtbgeom_Eh.notna() & GX.E_ald_dft_gxtbgeom_Eh.notna()).sum())
    labels = ["DFT geom-opt\n(13 hard EWG cases)",
              "g-xTB geom-opt\n(36 hard cases)",
              "DFT single-point on\ng-xTB geom (full 1%)"]
    rates = [0.0, 100.0, 100.0 * n_sp / len(GX)]
    cnts = ["0 / 13", "36 / 36", f"{n_sp} / {len(GX)}"]
    colors = ["#c0392b", "#2e7d32", "#2e7d32"]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    bars = ax.bar(labels, rates, color=colors, width=0.6)
    for b, c in zip(bars, cnts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, c,
                ha="center", fontsize=11)
    ax.set_ylabel("convergence rate  [%]"); ax.set_ylim(0, 112)
    ax.set_title("Conclusion 3 — DFT non-convergence is the real blocker,\n"
                 "but optimizing with g-xTB first lets the DFT single-point converge",
                 fontsize=11)
    fig.tight_layout(); out = FIGS / f"concl3_dft_convergence_blocker_{STAMP}.png"
    fig.savefig(out, dpi=150); plt.close(fig); return out


if __name__ == "__main__":
    for f in (fig_c1a, fig_c1b, fig_c2, fig_c3):
        print("wrote", f())
