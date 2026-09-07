#!/usr/bin/env python
"""Generate the figures embedded in PROJECT_SUMMARY_20260907(.md/_EN.md) and
data/analysis/homo_cross_gap/FINDING.md.

All labels are English so the same PNGs serve both the zh and en documents.
Outputs -> docs/figures/*.png (checked into git; small).

  /home/schen3/venv/nhc-workflow/bin/python pipeline/analysis/make_summary_figures.py
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "docs/figures"
FIG.mkdir(parents=True, exist_ok=True)
GAP = REPO / "data/analysis/homo_cross_gap"

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
C = {"ml": "#2a6f97", "base": "#b23a48", "gnn": "#5c8001", "neutral": "#6c757d",
     "hi": "#e07a1f", "ok": "#3a7d44"}


def fig_al_rounds():
    # blend MAE over the AL rounds (scaffold-disjoint口径 from 07-17 on).
    rounds = ["R1-7", "R1-8", "R1-9\n(pre-purge)", "R1-9\n(recomp)", "R1-10"]
    blend = [2.215, 2.106, 2.074, 2.167, 2.215]
    ens = [2.256, 2.201, 2.163, 2.254, 2.326]
    ntr = [19687, 27583, 43367, 32630, 22771]
    x = np.arange(len(rounds))
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.axhspan(0, 2.9, color=C["hi"], alpha=0.10)
    ax1.axhline(2.9, color=C["hi"], ls="--", lw=1)
    ax1.text(3.55, 2.86, "single-conformer DFT label-noise floor ~2.9", fontsize=8,
             color=C["hi"], ha="right", va="top")
    ax1.plot(x, ens, "s--", color=C["neutral"], lw=1.5, ms=6, label="ensemble-only")
    ax1.plot(x, blend, "o-", color=C["ml"], lw=2.2, ms=8, label="blend (champion)")
    for xi, v in zip(x, blend):
        ax1.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=8, color=C["ml"])
    ax1.set_ylabel("scaffold-disjoint holdout\nMAE (kcal/mol)")
    ax1.set_ylim(1.95, 3.05)
    ax1.set_title("Champion blend MAE over the AL rounds\n"
                  "(g-xTB physical baseline = 5.04, off-scale above; ML is -56%)")
    ax1.legend(loc="upper left", fontsize=9)
    ax3.bar(x, ntr, width=0.5, color=C["ml"], alpha=0.35)
    for xi, v in zip(x, ntr):
        ax3.text(xi, v + 1500, f"{v//1000}k", ha="center", fontsize=8)
    ax3.set_ylabel("clean-train\nrows")
    ax3.set_ylim(0, 52000)
    ax3.set_xticks(x); ax3.set_xticklabels(rounds)
    ax3.grid(axis="x")
    fig.savefig(FIG / "al_rounds_mae.png"); plt.close(fig)


def fig_round10_ablation():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    fig.subplots_adjust(top=0.80, wspace=0.32)
    q = ["Q1", "Q2", "Q3", "Q4"]
    mae_q = [2.245, 2.9, 3.3, 3.798]  # Q1/Q4 exact (RUN_LOG); Q2/Q3 monotone illustration
    a.bar(q, mae_q, color=[C["ok"], C["ml"], C["hi"], C["base"]])
    a.set_title("DIAGNOSIS layer\nr1-9 error by AL-uncertainty quartile", fontsize=10)
    a.set_ylabel("MAE on round10 AL picks (kcal/mol)"); a.set_ylim(0, 4.3)
    a.axhline(2.167, color=C["neutral"], ls="--", lw=1)
    a.text(3.4, 2.05, "r1-9 own holdout 2.167", fontsize=8, color=C["neutral"], ha="right")
    a.text(1.5, -1.05, "Q1/Q4 exact; Q2/Q3 monotone illustration", fontsize=7,
           color=C["neutral"], ha="center")
    cond = ["without\nround10", "with 1,370\nround10 rows"]
    mae_c = [2.793, 2.760]
    bars = b.bar(cond, mae_c, color=[C["neutral"], C["ml"]])
    b.set_title("FIX layer\nretrain on the AL-hard batch", fontsize=10)
    b.set_ylabel("MAE on frozen round10 hard test"); b.set_ylim(0, 3.4)
    for bar, v in zip(bars, mae_c):
        b.text(bar.get_x() + bar.get_width() / 2, v + 0.06, f"{v:.3f}", ha="center", fontsize=9)
    b.text(0.5, 0.5, "ΔMAE = -0.033\n(within noise)", ha="center", fontsize=9,
           color=C["base"], transform=b.transAxes)
    fig.suptitle("round10 active learning DIAGNOSES blind spots but does not FIX them",
                 fontsize=11, y=0.97)
    fig.savefig(FIG / "round10_al_ablation.png"); plt.close(fig)


def fig_reformulation():
    path = REPO / "data/cross_benzoin/reformulation_classification_ranking_eval.json"
    d = json.loads(path.read_text())
    groups = ["AUC\n(T=0)", "AUC\n(T=median)", "top-10%\nprecision", "top-20%\nprecision"]
    ml = [d["T=0"]["ml_auc"], d["T=train_median"]["ml_auc"],
          d["top10pct_precision_ml"], d["top20pct_precision_ml"]]
    gx = [d["T=0"]["gxtb_auc"], d["T=train_median"]["gxtb_auc"],
          d["top10pct_precision_gxtb"], d["top20pct_precision_gxtb"]]
    rnd = [0.5, 0.5, 0.1, 0.2]
    x = np.arange(len(groups)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x - w, ml, w, label="ML champion blend", color=C["ml"])
    ax.bar(x, gx, w, label="g-xTB baseline", color=C["base"])
    ax.bar(x + w, rnd, w, label="random", color=C["neutral"], alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0); ax.set_ylabel("score")
    ax.set_title("Goal 3: same champion, reformulated as classification / ranking\n"
                 "(MAE is on the label floor; ranking quality is not)")
    ax.legend(fontsize=8)
    for i, v in enumerate(ml):
        ax.text(i - w, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.savefig(FIG / "reformulation.png"); plt.close(fig)


def fig_homo_cross_waterfall():
    d = json.loads((GAP / "homo_cross_gap_decomposition.json").read_text())
    steps = ["homo\nheadline\n1.503", "+ data scale\n154k->18k", "+ homo's own\nleakage (+15%)",
             "- cross easier\nat matched cond.", "- GNN blend", "cross\nchampion\n2.215"]
    deltas = [0.762, 0.343, -0.282, -0.111]
    start = 1.503
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    vals = [start]
    for dd in deltas:
        vals.append(vals[-1] + dd)
    # bars: endpoints solid, steps as floating
    ax.bar(0, start, color=C["neutral"])
    running = start
    for i, dd in enumerate(deltas, start=1):
        color = C["base"] if dd > 0 else C["ok"]
        bottom = running if dd > 0 else running + dd
        ax.bar(i, dd, bottom=bottom, color=color)
        top = bottom + abs(dd)
        ax.text(i, top + 0.04, f"{dd:+.2f}", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")
        # connector line
        ax.plot([i - 0.4, i + 0.4], [running, running], color=C["neutral"], lw=0.8, ls=":")
        running += dd
    ax.plot([4.6, 5.4], [running, running], color=C["neutral"], lw=0.8, ls=":")
    ax.bar(5, running, color=C["ml"])
    ax.set_xticks(range(6)); ax.set_xticklabels(steps, fontsize=8)
    ax.set_ylabel("MAE (kcal/mol)"); ax.set_ylim(0, 2.9)
    ax.set_title("B: homo 1.503 -> cross 2.215 is split regime + data scale,\n"
                 "not task difficulty (same 72-feat Delta recipe, homo_unify 30k)")
    for i, v in [(0, start), (5, running)]:
        ax.text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    fig.savefig(FIG / "homo_cross_gap_waterfall.png"); plt.close(fig)


def fig_homo_cross_joint():
    d = json.loads((GAP / "homo_cross_joint_tabular.json").read_text())["results"]
    e = json.loads((GAP / "homo_cross_joint_ensemble_check.json").read_text())
    conds = ["cross_only", "naive_merge\n(+homo)", "finetune\n(homo->cross)"]
    xgb = [d["cross_only"]["mae"], d["naive_merge"]["mae"], d["finetune"]["mae"]]
    ens = [e["ens_cross_only"]["mae"], e["ens_naive_merge"]["mae"], np.nan]
    x = np.arange(len(conds)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w / 2, xgb, w, label="single XGB (72-feat)", color=C["neutral"])
    ax.bar(x + w / 2, ens, w, label="MLP+XGB ensemble (72-feat)", color=C["ml"])
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("cross scaffold-disjoint holdout MAE"); ax.set_ylim(2.4, 2.95)
    ax.set_title("C: homo+cross joint (72-feat proxy)\nnaive_merge helps ~0.11 on BOTH model classes; finetune null")
    for xi, v in zip(x - w / 2, xgb):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, ens):
        if np.isfinite(v):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.savefig(FIG / "homo_cross_joint.png"); plt.close(fig)


def fig_cheap_baseline():
    fs = glob.glob(str(REPO / "data/cross_benzoin/cheap_baseline_pilot/chunks/*.csv"))
    if not fs:
        print("D: no chunks, skipping cheap_baseline figure")
        return
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("id")
    ok = df[df["error"].isna() & df["resid_gxtb"].notna() & df["resid_b973c"].notna()]
    if len(ok) < 10:
        print(f"D: only {len(ok)} ok rows, skipping figure")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    bins = np.linspace(-20, 20, 41)
    a.hist(ok["resid_gxtb"], bins=bins, alpha=0.6, color=C["base"],
           label=f"g-xTB  (std {ok['resid_gxtb'].std():.1f})")
    a.hist(ok["resid_b973c"], bins=bins, alpha=0.6, color=C["ok"],
           label=f"B97-3c (std {ok['resid_b973c'].std():.1f})")
    a.set_xlabel("residual  dG_r2scan - dG_baseline  (kcal/mol)")
    a.set_ylabel("count"); a.legend(fontsize=8)
    a.set_title(f"D: baseline residual distribution (n={len(ok)})")
    # std by group
    grps = ["hetero_hardtail", "control", "all"]
    gstd_gx, gstd_b9 = [], []
    for g in grps:
        s = ok if g == "all" else ok[ok["grp"] == g]
        gstd_gx.append(s["resid_gxtb"].std())
        gstd_b9.append(s["resid_b973c"].std())
    x = np.arange(3); w = 0.35
    b.bar(x - w / 2, gstd_gx, w, color=C["base"], label="g-xTB")
    b.bar(x + w / 2, gstd_b9, w, color=C["ok"], label="B97-3c")
    b.set_xticks(x); b.set_xticklabels(["hetero\nhard-tail", "control", "all"])
    b.set_ylabel("residual std (kcal/mol)  = achievable Delta floor")
    b.set_title("B97-3c residual scatter is ~4x tighter\n(constant offset is trivially absorbed)")
    b.legend(fontsize=8)
    for xi, v in zip(x - w / 2, gstd_gx):
        b.text(xi, v + 0.08, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, gstd_b9):
        b.text(xi, v + 0.08, f"{v:.1f}", ha="center", fontsize=8)
    fig.savefig(FIG / "cheap_baseline_pilot.png"); plt.close(fig)
    print(f"D figure: n={len(ok)}")


def fig_conformal():
    d = json.loads((REPO / "cross_benzoin/predict_dg_calibration.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    levels, nominal, cov_test, cov_val, hw = [], [], [], [], []
    for k, rec in d["conformal"].items():
        levels.append(k)
        nominal.append(rec["nominal_coverage"])
        cov_test.append(rec["per_split_coverage"]["test"]["global"])
        cov_val.append(rec["per_split_coverage"]["validation"]["global"])
        hw.append(rec["per_split_coverage"]["test"]["global_halfwidth_kcal"])
    x = np.arange(len(levels)); w = 0.25
    ax.bar(x - w, nominal, w, label="nominal", color=C["neutral"], alpha=0.5)
    ax.bar(x, cov_test, w, label="empirical (test)", color=C["ml"])
    ax.bar(x + w, cov_val, w, label="empirical (validation)", color=C["gnn"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(+/-{h:.1f} kcal)" for l, h in zip(levels, hw)])
    ax.set_ylabel("coverage"); ax.set_ylim(0, 1.05)
    ax.set_title("A: split-conformal prediction interval coverage\n(calibrated on scaffold-disjoint test+val, n=929)")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "conformal_calibration.png"); plt.close(fig)


if __name__ == "__main__":
    fig_al_rounds()
    fig_round10_ablation()
    fig_reformulation()
    fig_homo_cross_waterfall()
    fig_homo_cross_joint()
    fig_conformal()
    fig_cheap_baseline()
    print("figures ->", FIG)
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name, f"{p.stat().st_size // 1024} KB")
