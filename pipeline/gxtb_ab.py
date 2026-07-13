#!/usr/bin/env python3
"""
A/B: GFN2 vs g-xTB as the Δ-learning BASELINE  (experiment, 2026-06-22).

Merges `dG_gxtb_kcal` (from pipeline/compute/gxtb_baseline.py, COSMO/DMSO, on the
funnel_v3 geometry) into the featurize table and compares, on the SAME matched
molecule set, the Δ-model trained on each baseline:

    pred_DFT = baseline + model(features, target - baseline)
    metric   = repeated-KFold out-of-fold MAE/RMSE/r vs dG_orca_kcal

Reported per baseline: (1) RAW baseline error |baseline - DFT| (no ML), and
(2) Δ-model CV error. Isolates the baseline swap: same rows, same features, same
model/CV; the ONLY change is which semiempirical ΔG is the baseline (and feature).
The corr-MAD outlier QC is OFF here (it is baseline-dependent and would compare
different row sets); scope + reactive + |ΔG|<=45 filters still apply.

Usage:
  python gxtb_ab.py --gxtb data/raw/gxtb_baseline/chunks/chunk_*.csv \
                    [--model xgb] [--folds 5 --repeats 4]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "pipeline" / "compute"))

import delta_core as dc

TARGET = "dG_orca_kcal"
XTB = "dG_xtb_kcal"
GXTB = "dG_gxtb_kcal"


def _load_merged(gxtb_glob: str) -> pd.DataFrame:
    df = pd.read_parquet(dc.DEFAULT_FEATURIZE_PARQUET)
    files = sorted(glob.glob(gxtb_glob))
    if not files:
        raise FileNotFoundError(f"no g-xTB baseline files match {gxtb_glob}")
    g = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = g[["index", GXTB]].dropna(subset=[GXTB])
    g[GXTB] = pd.to_numeric(g[GXTB], errors="coerce")
    print(f"  merged g-xTB baseline: {len(files)} files, {g[GXTB].notna().sum()} values")
    return df.merge(g, on="index", how="left")


def _scope_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Reactive + aromatic-CHO-scope + |ΔG|<=45, matching delta_core (sans corr-MAD)."""
    full = df.copy()
    try:
        from filter_smiles import classify
        full = full[~full["SMILES"].map(classify).isin(dc._REACTIVE_REASONS)]
    except Exception as e:
        print("  (reactive filter skipped:", e, ")")
    try:
        from cho_category import cho_class
        full = full[full["SMILES"].map(cho_class).isin(dc.CHO_SCOPE)]
    except Exception as e:
        print("  (scope filter skipped:", e, ")")
    if "n_CHO" in full:
        full = full[pd.to_numeric(full["n_CHO"], errors="coerce") == 1]
    for c in (TARGET, XTB, GXTB):
        full[c] = pd.to_numeric(full[c], errors="coerce")
    full = full[full[TARGET].abs() <= 45]
    return full


def _features(df: pd.DataFrame, baseline: str) -> list[str]:
    """Numeric descriptor columns + the chosen baseline; BOTH raw baselines excluded
    from descriptors so the only baseline-derived feature is `baseline` itself."""
    drop = set(dc.ID_COLS) | set(dc.FILTER_COLS) | {TARGET, XTB, GXTB} | set(dc.LABEL_EXTRA)
    feats = [c for c in df.columns
             if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return feats + [baseline]


def _cv(df: pd.DataFrame, baseline: str, model_kind: str, folds: int, repeats: int):
    feats = _features(df, baseline)
    X = df[feats].copy()
    X = X.fillna(X.median(numeric_only=True))
    base = df[baseline].to_numpy(float)
    dft = df[TARGET].to_numpy(float)
    y = dft - base                      # correction to learn
    oof = np.zeros_like(dft)
    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    counts = np.zeros_like(dft)
    for tr, te in rkf.split(X):
        m = dc.build_model(model_kind)
        m.fit(X.iloc[tr], y[tr])
        oof[te] += base[te] + m.predict(X.iloc[te])
        counts[te] += 1
    oof /= np.maximum(counts, 1)
    return dict(
        n=len(df),
        raw_mae=mean_absolute_error(dft, base),
        raw_bias=float(np.mean(base - dft)),
        raw_r=float(np.corrcoef(base, dft)[0, 1]),
        cv_mae=mean_absolute_error(dft, oof),
        cv_rmse=float(np.sqrt(mean_squared_error(dft, oof))),
        cv_r=float(r2_score(dft, oof)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gxtb", default=str(REPO / "data/raw/gxtb_baseline/chunks/chunk_*.csv"))
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()

    df = _load_merged(args.gxtb)
    df = _scope_filter(df)
    matched = df[df[XTB].notna() & df[GXTB].notna() & df[TARGET].notna()].copy()
    print(f"\nmatched rows (both baselines + DFT, in-scope): {len(matched)}")
    print(f"  g-xTB coverage of in-scope labeled: "
          f"{matched[GXTB].notna().sum()} / {df[XTB].notna().sum()}")

    print(f"\n{'baseline':>8} | {'raw MAE':>7} {'raw bias':>8} {'raw r':>6} | "
          f"{'CV MAE':>6} {'CV RMSE':>7} {'CV R2':>6}")
    print("-" * 64)
    rows = []
    for name, col in [("GFN2", XTB), ("g-xTB", GXTB)]:
        r = _cv(matched, col, args.model, args.folds, args.repeats)
        rows.append((name, r))
        print(f"{name:>8} | {r['raw_mae']:7.2f} {r['raw_bias']:8.2f} {r['raw_r']:6.3f} | "
              f"{r['cv_mae']:6.2f} {r['cv_rmse']:7.2f} {r['cv_r']:6.3f}")
    print()
    g_xtb, g_gxtb = rows[0][1], rows[1][1]
    print(f"Δ(CV MAE) g-xTB − GFN2 = {g_gxtb['cv_mae'] - g_xtb['cv_mae']:+.3f} kcal "
          f"(raw baseline MAE {g_xtb['raw_mae']:.1f} -> {g_gxtb['raw_mae']:.1f})")


if __name__ == "__main__":
    main()
