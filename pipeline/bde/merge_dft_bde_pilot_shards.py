#!/usr/bin/env python
"""Combine dft_bde_pilot.py --n-shards outputs into one table and report the g-xTB-vs-DFT
and ALFABET-vs-DFT arbitration stats at the scaled-up sample size (Phase-1 next-step #3,
2026-07-15) -- the n=25 pilot found g-xTB tracks DFT much better than ALFABET (r=0.63 vs
0.44, memory `dft-arbitration-result`), this checks whether that holds at n~100 and
quantifies g-xTB's own MAE-vs-DFT gap (candidate noise floor for B4/B5).

Usage:
  python merge_dft_bde_pilot_shards.py --glob "/tmp/dft_bde_pilot_shard*.csv" \
      --out /tmp/dft_bde_pilot_n100.csv --report-out /tmp/dft_bde_pilot_n100_report.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    assert files, f"no files matched {args.glob}"
    df = pd.concat([pd.read_csv(f, dtype={"id": str}) for f in files], ignore_index=True)
    df = df.drop_duplicates("id")
    df.to_csv(args.out, index=False)
    n_total = len(df)
    ok = df.dropna(subset=["bde_dft_kcal"])
    n_ok = len(ok)
    print(f"merged {len(files)} shards: {n_ok}/{n_total} succeeded", flush=True)

    def stats(pred_col):
        y_true, y_pred = ok["bde_dft_kcal"].to_numpy(), ok[pred_col].to_numpy()
        r, _ = pearsonr(y_true, y_pred)
        rho, _ = spearmanr(y_true, y_pred)
        return {
            "pearson_r": float(r), "spearman_rho": float(rho),
            "MAE_vs_dft": float(mean_absolute_error(y_true, y_pred)),
            "R2_vs_dft": float(r2_score(y_true, y_pred)),
            "mean_signed_error": float(np.mean(y_pred - y_true)),
        }

    report = {
        "n_shards": len(files), "n_total": n_total, "n_succeeded": n_ok,
        "bde_dft_range": [float(ok["bde_dft_kcal"].min()), float(ok["bde_dft_kcal"].max())],
        "gxtb_vs_dft": stats("bde_gxtb_kcal"),
        "alfabet_vs_dft": stats("bde_alfabet_kcal"),
    }
    print(json.dumps(report, indent=2))
    Path(args.report_out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out} and {args.report_out}")


if __name__ == "__main__":
    main()
