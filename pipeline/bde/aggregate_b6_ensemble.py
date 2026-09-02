#!/usr/bin/env python
"""Aggregate the B6 5-seed deep ensemble (submit_b6_deep_ensemble.sh) into a
mean/std prediction table + an epistemic-uncertainty triage summary.

For each task (aldehydes, products):
  - outer-merge the per-seed test predictions on `id`
  - ensemble prediction = mean over seeds; epistemic sigma = std over seeds
  - report ensemble MAE/RMSE/R2/spearman vs the mean single-seed number
  - simulate the inference-time "route the least-confident to DFT" rule at a few
    coverage levels and report the MAE on the kept (not-routed) subset -- if the
    sigma signal is any good, kept-MAE drops well below the overall MAE.

Usage:
  python aggregate_b6_ensemble.py \
    --ens-dir runs/logs/scaffold_disjoint_bde/ensemble [--which aldehydes products]
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def load_task(ens_dir: Path, which: str):
    files = sorted(glob.glob(str(ens_dir / f"{which}_seed*_pred.csv")))
    if not files:
        return None, []
    merged = None
    seeds = []
    for f in files:
        m = re.search(r"_seed(\d+)_pred\.csv$", f)
        seed = int(m.group(1)) if m else len(seeds)
        seeds.append(seed)
        d = pd.read_csv(f)[["id", "y_true", "y_pred"]].rename(
            columns={"y_pred": f"pred_s{seed}", "y_true": f"true_s{seed}"})
        merged = d if merged is None else merged.merge(d, on="id", how="outer")
    pred_cols = [c for c in merged.columns if c.startswith("pred_s")]
    true_cols = [c for c in merged.columns if c.startswith("true_s")]
    # y_true should agree across seeds (same split); take the row-wise first non-null
    merged["y_true"] = merged[true_cols].bfill(axis=1).iloc[:, 0]
    merged["y_pred_mean"] = merged[pred_cols].mean(axis=1)
    merged["y_pred_std"] = merged[pred_cols].std(axis=1, ddof=1)
    merged["abs_err"] = (merged["y_pred_mean"] - merged["y_true"]).abs()
    out = merged[["id", "y_true", "y_pred_mean", "y_pred_std", "abs_err"] + pred_cols]
    return out.dropna(subset=["y_true"]).reset_index(drop=True), seeds


def metrics(y, p):
    return dict(
        n=int(len(y)),
        mae=float(mean_absolute_error(y, p)),
        rmse=float(root_mean_squared_error(y, p)),
        r2=float(r2_score(y, p)),
        spearman=float(spearmanr(y, p).statistic),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens-dir", type=Path, required=True)
    ap.add_argument("--which", nargs="+", default=["aldehydes", "products"])
    ap.add_argument("--route-coverages", nargs="+", type=float,
                    default=[0.95, 0.90, 0.80],
                    help="fraction KEPT by ML; the rest (highest sigma) route to DFT")
    args = ap.parse_args()

    summary = {}
    for which in args.which:
        df, seeds = load_task(args.ens_dir, which)
        if df is None:
            print(f"[{which}] no per-seed pred files in {args.ens_dir} -- skip")
            continue
        pred_cols = [c for c in df.columns if c.startswith("pred_s")]
        per_seed = [metrics(df["y_true"], df[c])["mae"] for c in pred_cols]
        ens = metrics(df["y_true"], df["y_pred_mean"])

        # correlation of epistemic sigma with the actual error (want > 0)
        sig_err_r = float(spearmanr(df["y_pred_std"], df["abs_err"]).statistic)

        routing = []
        for cov in args.route_coverages:
            thr = df["y_pred_std"].quantile(cov)
            kept = df[df["y_pred_std"] <= thr]
            routed = df[df["y_pred_std"] > thr]
            routing.append(dict(
                keep_fraction=round(float(len(kept) / len(df)), 4),
                sigma_threshold=float(thr),
                kept_mae=float(kept["abs_err"].mean()),
                routed_mae=float(routed["abs_err"].mean()) if len(routed) else None,
                routed_n=int(len(routed)),
            ))

        s = dict(
            seeds=seeds,
            per_seed_mae=per_seed,
            per_seed_mae_mean=float(np.mean(per_seed)),
            ensemble=ens,
            ensemble_gain_vs_mean_seed=float(np.mean(per_seed) - ens["mae"]),
            sigma_vs_abserr_spearman=sig_err_r,
            mean_sigma=float(df["y_pred_std"].mean()),
            routing=routing,
        )
        summary[which] = s
        out_csv = args.ens_dir / f"{which}_ensemble_pred.csv"
        df.to_csv(out_csv, index=False)
        print(f"[{which}] seeds={seeds} per-seed MAE {np.mean(per_seed):.3f} -> "
              f"ensemble MAE {ens['mae']:.3f} (R2 {ens['r2']:.3f}); "
              f"sigma~|err| spearman {sig_err_r:.3f}")
        for r in routing:
            print(f"    keep {r['keep_fraction']*100:.0f}%  kept_MAE {r['kept_mae']:.3f}"
                  f"   (routed {r['routed_n']} @ MAE {r['routed_mae']})")
        print(f"    wrote {out_csv}")

    out_json = args.ens_dir / "ensemble_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
