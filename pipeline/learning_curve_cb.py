#!/usr/bin/env python
"""Learning curve — how many DFT labels are 'enough' for the cb Δ-model.

Silhouette k-means can't size the label set (the v6 chemical space is continuous;
silhouette ~0.05 at all k). The principled answer is a learning curve: hold out a fixed
TEST set, then train on increasing subsets of the remaining data and watch test MAE vs n.
Where it plateaus = the point past which extra DFT labels stop paying off.

Geometry-consistent: funnel_v3 r2SCAN-3c labels + g-xTB baseline + product descriptors,
ALL CHO categories (no aromatic filter). One standalone figure + a CSV/markdown summary.

  python pipeline/learning_curve_cb.py --parquet data/featurize_cb_homo_train_gxtb.parquet
"""
from __future__ import annotations
import argparse, datetime, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

REPO = Path("/scratch-shared/schen3/benzoin-dg")
OUT = REPO / "data/analysis/learning_curve_cb"
NON_FEAT = {"c", "SMILES", "index", "PubChem_CID", "dG_orca_kcal", "dG_orca_shermo_kcal",
            "dG_shermo_kcal", "aldehyde_smiles", "error"}
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# fixed, sensible xgb config (held constant so the curve reflects DATA SIZE, not tuning)
PARAMS = dict(n_estimators=800, max_depth=5, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=3,
              tree_method="hist", n_jobs=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(REPO / "data/featurize_cb_homo_train_gxtb.parquet"))
    ap.add_argument("--target", default="dG_orca_kcal")
    ap.add_argument("--baseline", default="dG_xtb_kcal")
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    df = df[df[args.target].notna() & df[args.baseline].notna()].reset_index(drop=True)
    feats = [c for c in df.columns if c not in NON_FEAT and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    base = df[args.baseline].to_numpy(); dft = df[args.target].to_numpy(); y = dft - base
    N = len(df)
    print(f"rows={N} features={len(feats)}")

    # fixed held-out test
    pool, test = train_test_split(np.arange(N), test_size=args.test_frac, random_state=args.seed)
    n_pool = len(pool)
    grid = sorted({int(n) for n in [200, 400, 600, 900, 1200, 1600, 2000, 2500, n_pool]
                   if n <= n_pool})
    rng = np.random.default_rng(args.seed)
    rows = []
    for n in grid:
        maes, bmaes = [], []
        for r in range(args.repeats):
            tr = rng.choice(pool, size=n, replace=False)
            m = xgb.XGBRegressor(random_state=args.seed + r, **PARAMS)
            m.fit(X.iloc[tr], y[tr])
            pred = base[test] + m.predict(X.iloc[test])
            maes.append(mean_absolute_error(dft[test], pred))
            bmaes.append(mean_absolute_error(dft[test], base[test]))   # g-xTB baseline
        rows.append(dict(n=n, mae=np.mean(maes), mae_std=np.std(maes),
                         baseline_mae=np.mean(bmaes)))
        print(f"  n={n:5d}  test MAE={np.mean(maes):.3f} ± {np.std(maes):.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"{TS}_learning_curve.csv", index=False)

    plt.figure(figsize=(7.4, 5.0))
    plt.errorbar(res.n, res.mae, yerr=res.mae_std, marker="o", capsize=3, label="Δ-model (g-xTB + product desc)")
    plt.axhline(res.baseline_mae.mean(), color="grey", ls="--", label=f"g-xTB baseline ({res.baseline_mae.mean():.2f})")
    plt.xlabel("# training molecules"); plt.ylabel(f"held-out test MAE (kcal/mol, {len(test)} mols)")
    plt.title(f"Learning curve — cb Δ-model (r2SCAN-3c labels, all categories, N={N})")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / f"{TS}_learning_curve.png", dpi=130); plt.close()

    # plateau heuristic: first n where adding the next step改善 < 0.05 kcal
    res = res.sort_values("n").reset_index(drop=True)
    plateau = None
    for i in range(1, len(res)):
        if res.mae[i-1] - res.mae[i] < 0.05:
            plateau = int(res.n[i-1]); break
    print(f"\nfull-data test MAE={res.mae.iloc[-1]:.3f}  plateau≈{plateau}  baseline={res.baseline_mae.mean():.2f}")
    print(f"wrote {OUT}/{TS}_learning_curve.png + .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
