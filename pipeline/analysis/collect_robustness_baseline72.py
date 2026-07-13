#!/usr/bin/env python
"""Aggregate robustness_baseline72.py per-task JSONs into a summary report + plots.
Answers: overfitting (train vs val vs test gap), seed sensitivity (5 reshuffled 70:20:10
holdouts), and cross-validation stability (5-fold CV) for baseline_72.
"""
import json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

H = "/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6"
IN = Path(f"{H}/viz_gxtb_20260625/robustness")
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d")


def savefig(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print("wrote", name, flush=True)


def main():
    holdouts = sorted(IN.glob("result_holdout_seed*.json"))
    cvs = sorted(IN.glob("result_cv_fold*.json"))
    hres = [json.load(open(f)) for f in holdouts]
    cres = [json.load(open(f)) for f in cvs]
    print(f"loaded {len(hres)} holdout seeds, {len(cres)} cv folds", flush=True)

    # ── 1) overfitting: train/val/test MAE per holdout seed ─────────────
    seeds = [r["seed"] for r in hres]
    tr_mae = [r["train"]["mae"] for r in hres]
    va_mae = [r["val"]["mae"] for r in hres]
    te_mae = [r["test"]["mae"] for r in hres]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seeds)); w = 0.25
    ax.bar(x - w, tr_mae, w, label="train", color="#2171b5")
    ax.bar(x, va_mae, w, label="val", color="#41ab5d")
    ax.bar(x + w, te_mae, w, label="test", color="#cb181d")
    ax.set_xticks(x); ax.set_xticklabels([f"seed{s}" for s in seeds])
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("baseline_72: train/val/test MAE across reshuffled 70:20:10 holdouts")
    ax.legend()
    savefig(fig, f"60_overfit_train_val_test_{TAG}.png")

    # ── 2) seed sensitivity: test MAE distribution ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([f"seed{s}" for s in seeds], te_mae, color="#cb181d")
    ax.axhline(np.mean(te_mae), color="k", ls="--", lw=1,
              label=f"mean={np.mean(te_mae):.3f} std={np.std(te_mae):.3f}")
    ax.set_ylabel("test MAE (kcal/mol)")
    ax.set_title("baseline_72: test MAE across 5 reshuffled seeds (same 70:20:10 ratio)")
    ax.legend()
    savefig(fig, f"61_seed_sensitivity_{TAG}.png")

    # ── 3) cross-validation stability ────────────────────────────────────
    if cres:
        folds = [r["fold"] for r in cres]
        cv_te_mae = [r["test"]["mae"] for r in cres]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar([f"fold{f}" for f in folds], cv_te_mae, color="#238b45")
        ax.axhline(np.mean(cv_te_mae), color="k", ls="--", lw=1,
                  label=f"mean={np.mean(cv_te_mae):.3f} std={np.std(cv_te_mae):.3f}")
        ax.axhline(np.mean(te_mae), color="#cb181d", ls=":", lw=1.5,
                  label=f"holdout mean={np.mean(te_mae):.3f}")
        ax.set_ylabel("test MAE (kcal/mol)")
        ax.set_title(f"baseline_72: {len(cres)}-fold CV test MAE vs holdout mean")
        ax.legend()
        savefig(fig, f"62_cv_stability_{TAG}.png")

    # ── written report ────────────────────────────────────────────────────
    rep = OUT / f"REPORT_robustness_baseline72_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# baseline_72 overfitting / robustness / reproducibility ({TAG})\n\n")
        fh.write("## Overfitting (train vs val vs test, per reshuffled seed)\n\n")
        fh.write("| seed | train MAE | val MAE | test MAE | test-train gap |\n|---|---|---|---|---|\n")
        for r in hres:
            fh.write(f"| {r['seed']} | {r['train']['mae']:.3f} | {r['val']['mae']:.3f} | "
                     f"{r['test']['mae']:.3f} | {r['test']['mae']-r['train']['mae']:.3f} |\n")
        fh.write(f"\nmean gap = {np.mean([r['test']['mae']-r['train']['mae'] for r in hres]):.3f}\n\n")
        fh.write("## Seed sensitivity (5x reshuffled 70:20:10, same ratios)\n\n")
        fh.write(f"test MAE = {np.mean(te_mae):.3f} +/- {np.std(te_mae):.3f} "
                 f"(range {min(te_mae):.3f}-{max(te_mae):.3f})\n\n")
        if cres:
            fh.write(f"## {len(cres)}-fold cross-validation\n\n")
            fh.write("| fold | train MAE | val MAE | test MAE |\n|---|---|---|---|\n")
            for r in cres:
                fh.write(f"| {r['fold']} | {r['train']['mae']:.3f} | {r['val']['mae']:.3f} | {r['test']['mae']:.3f} |\n")
            fh.write(f"\nCV test MAE = {np.mean(cv_te_mae):.3f} +/- {np.std(cv_te_mae):.3f} "
                     f"vs holdout mean {np.mean(te_mae):.3f}\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
