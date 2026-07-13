#!/usr/bin/env python3
"""
Prediction-accuracy parity figures for the benzoin Δ-model, split into separate
files (xTB baseline / Δ-learning), with the large-error molecules drawn on the
Δ-learning panel, plus a "drop-outliers-and-retrain" diagnostic.

Reads the saved CV predictions (runs/data/cv_predictions.csv) for the existing
plots, and re-runs delta_core CV on the cleaned subset for the retrain figure.

  python pipeline/plot_parity_panels.py [--res-cut 8.0]

Outputs (runs/figs/):
  parity_xtb.png            xTB baseline, single panel
  parity_delta.png          Δ-learning, single panel + 4 worst-molecule structures
  parity_delta_clean.png    Δ-learning after dropping |residual|>cut and retraining
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE, r2_score

import delta_core as dc

FIGS = dc.REPO_ROOT / "runs/figs"
XTB_COL, DEL_COL, OUT_COL = "#4C72B0", "#DD8452", "#C44E52"


def _stats(t, p):
    return MAE(t, p), float(np.sqrt(MSE(t, p))), r2_score(t, p)


def _parity(ax, y, p, title, color, lims):
    ax.plot(lims, lims, "k--", lw=1, zorder=1)
    ax.scatter(y, p, s=14, alpha=0.45, color=color, zorder=2, edgecolors="none")
    ax.set_xlim(*lims); ax.set_ylim(*lims); ax.set_aspect("equal")
    m, r, r2 = _stats(y, p)
    ax.set_xlabel("DFT r2SCAN-3c  ΔG (kcal/mol)")
    ax.set_ylabel("predicted ΔG (kcal/mol)")
    ax.set_title(f"{title}\nMAE {m:.2f}   RMSE {r:.2f}   R² {r2:.2f}", fontsize=11)
    ax.grid(ls=":", alpha=0.4)
    return m, r, r2


def fig_xtb(y, xtb):
    lims = (min(y.min(), xtb.min()) - 1, max(y.max(), xtb.max()) + 1)
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    _parity(ax, y, xtb, "xTB baseline (no ML)", XTB_COL, lims)
    fig.tight_layout(); fig.savefig(FIGS / "parity_xtb.png", dpi=140); plt.close(fig)
    print("saved parity_xtb.png")


def fig_delta(d, y, pred, res_cut):
    res = pred - y; ares = np.abs(res)
    # Focused, equal-aspect axis: cover all Δ-learning points (don't clip outliers),
    # but the body is tight because predictions don't have xTB's big offset.
    lo = float(np.floor(min(y.min(), pred.min()))) - 1
    hi = float(np.ceil(max(y.max(), pred.max()))) + 1
    lims = (lo, hi)
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    _parity(ax, y, pred, f"Δ-learning prediction  (n={len(y)})", DEL_COL, lims)
    m8 = ares > res_cut
    ax.scatter(y[m8], pred[m8], s=46, facecolors="none", edgecolors=OUT_COL, lw=1.5,
               zorder=3, label=f"|residual| > {res_cut:g}  ({int(m8.sum())} mols)")
    ax.legend(loc="lower right", fontsize=9)

    # Pick 4 worst-residual molecules that are small enough to render cleanly.
    cand = d.assign(res=res, ares=ares)
    cand["nat"] = cand["SMILES"].map(
        lambda s: (Chem.MolFromSmiles(s).GetNumHeavyAtoms()
                   if Chem.MolFromSmiles(s) else 999))
    pick = cand[cand["nat"] <= 32].sort_values("ares", ascending=False).head(4)
    # 4 corners in axes fraction for the structure insets
    corners = [(0.20, 0.84), (0.80, 0.84), (0.20, 0.20), (0.80, 0.20)]
    for (_, row), (fx, fy) in zip(pick.iterrows(), corners):
        mol = Chem.MolFromSmiles(row["SMILES"])
        img = Draw.MolToImage(mol, size=(150, 150))
        ab = AnnotationBbox(
            OffsetImage(np.asarray(img), zoom=0.62),
            xy=(row["dG_orca_kcal"], row["dG_pred"]), xycoords="data",
            xybox=(fx, fy), boxcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=OUT_COL, lw=1.3),
            bboxprops=dict(edgecolor=OUT_COL, lw=1.0), pad=0.2, frameon=True)
        ax.add_artist(ab)
        ax.annotate(f"idx {int(row['index'])}  res {row['res']:+.1f}",
                    xy=(fx, fy), xycoords="axes fraction",
                    xytext=(0, -34), textcoords="offset points",
                    ha="center", fontsize=7.5, color=OUT_COL)
    fig.tight_layout(); fig.savefig(FIGS / "parity_delta.png", dpi=140); plt.close(fig)
    print(f"saved parity_delta.png  (insets: idx {list(pick['index'].astype(int))})")


def fig_delta_density(d, y, pred, res_cut):
    """Same Δ-learning parity, but hexbin density + marginal histograms so the
    crowded body (50% of points in a ~5 kcal window) is readable."""
    res = pred - y; ares = res_cut and np.abs(pred - y)
    lo = float(np.floor(min(y.min(), pred.min()))) - 1
    hi = float(np.ceil(max(y.max(), pred.max()))) + 1
    fig = plt.figure(figsize=(7.6, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                          wspace=0.03, hspace=0.03)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    hb = ax.hexbin(y, pred, gridsize=42, cmap="YlOrBr", mincnt=1,
                   extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    m8 = np.abs(res) > res_cut
    ax.scatter(y[m8], pred[m8], s=28, facecolors="none", edgecolors=OUT_COL, lw=1.2,
               label=f"|residual| > {res_cut:g}  ({int(m8.sum())})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    m, r, r2 = _stats(y, pred)
    ax.set_xlabel("DFT r2SCAN-3c  ΔG (kcal/mol)")
    ax.set_ylabel("predicted ΔG (kcal/mol)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(ls=":", alpha=0.4)
    bins = np.linspace(lo, hi, 46)
    axt.hist(y, bins=bins, color=DEL_COL, alpha=0.8)
    axr.hist(pred, bins=bins, orientation="horizontal", color=DEL_COL, alpha=0.8)
    for a in (axt, axr):
        a.axis("off")
    axt.set_title(f"Δ-learning parity (density)   MAE {m:.2f}  RMSE {r:.2f}  R² {r2:.2f}",
                  fontsize=11)
    fig.colorbar(hb, ax=axr, fraction=0.5, pad=0.05, label="count")
    fig.savefig(FIGS / "parity_delta_density.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved parity_delta_density.png")


def fig_clean(d, res_cut):
    """Drop |residual|>cut molecules and re-run CV — the 'noise-cleaned ceiling'."""
    tbl = dc.load_training_table()
    res = (d.set_index("index")["dG_pred"] - d.set_index("index")["dG_orca_kcal"])
    ares = res.abs().reindex(tbl.df["index"].to_numpy()).to_numpy()
    keep = ares <= res_cut
    sub = dc.TrainTable(
        df=tbl.df[keep].reset_index(drop=True), feats=tbl.feats, target=tbl.target,
        X=tbl.X[keep].reset_index(drop=True), y=tbl.y[keep],
        dG_xtb=tbl.dG_xtb[keep], dG_dft=tbl.dG_dft[keep], medians=tbl.medians)
    delta, base, oof = dc.cv_evaluate(sub, "xgb")
    n_drop = int((~keep).sum())

    y = sub.dG_dft
    lo = float(np.floor(min(y.min(), oof.min()))) - 1
    hi = float(np.ceil(max(y.max(), oof.max()))) + 1
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _parity(ax, y, oof,
            f"Δ-learning, outliers removed  (n={len(sub.df)}, |res|≤{res_cut:g})",
            "#55A868", (lo, hi))
    fname = f"parity_delta_clean_res{res_cut:g}.png"
    fig.tight_layout(); fig.savefig(FIGS / fname, dpi=140); plt.close(fig)
    print(f"saved {fname}")
    return dict(cut=res_cut, dropped=n_drop, n=len(sub.df), **delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-cut", type=float, default=8.0,
                    help="drop molecules with |CV residual| above this (kcal/mol)")
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(dc.REPO_ROOT / "runs/data/cv_predictions.csv")
    y = d["dG_orca_kcal"].to_numpy()
    xtb = d["dG_xtb_kcal"].to_numpy()
    pred = d["dG_pred"].to_numpy()
    fig_xtb(y, xtb)
    fig_delta(d, y, pred, a.res_cut)
    fig_delta_density(d, y, pred, a.res_cut)
    rows = [fig_clean(d, c) for c in (5.0, 8.0)]
    print(f"\n{'cut':>5} {'dropped':>8} {'n':>6} {'MAE':>6} {'RMSE':>6} {'R2':>6}")
    print(f"{'all':>5} {0:>8} {len(d):>6} {2.23:>6.2f} {3.02:>6.2f} {0.58:>6.2f}")
    for r in rows:
        print(f"{r['cut']:>5g} {r['dropped']:>8} {r['n']:>6} "
              f"{r['MAE']:>6.2f} {r['RMSE']:>6.2f} {r['R2']:>6.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
