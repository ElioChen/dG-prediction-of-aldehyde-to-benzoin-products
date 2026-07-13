#!/usr/bin/env python3
"""
Learning curve for the benzoin Δ-learning tree model: CV MAE vs training-set
size, computed by sub-sampling the full r2SCAN-3c table and running the same
repeated-K-fold CV used everywhere else. GNN points (expensive, GPU-only) are
overlaid from values recorded in pipeline/docs/gnn_delta.md.

Usage
  python pipeline/learning_curve.py                       # tuned xgb (runs/models/metadata.json)
  python pipeline/learning_curve.py --model xgb --reps 5  # custom
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import delta_core as dc


def cv_subset(tbl: dc.TrainTable, idx: np.ndarray, kind: str, params: dict,
              folds: int, repeats: int, seed: int):
    """Repeated-K-fold CV restricted to rows `idx` (a sub-sample of the table)."""
    from dataclasses import replace
    sub = dc.TrainTable(
        df=tbl.df.iloc[idx], feats=tbl.feats, target=tbl.target,
        X=tbl.X.iloc[idx].reset_index(drop=True), y=tbl.y[idx],
        dG_xtb=tbl.dG_xtb[idx], dG_dft=tbl.dG_dft[idx], medians=tbl.medians,
    )
    delta, base, _ = dc.cv_evaluate(sub, kind, params, folds, repeats, seed)
    return delta, base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="xgb", choices=["xgb", "rf", "gbt"])
    ap.add_argument("--params", default=None,
                    help="JSON params; default reads runs/models/metadata.json")
    ap.add_argument("--sizes", default="200,400,600,800,1000,1200,1427")
    ap.add_argument("--seeds", type=int, default=3, help="sub-sample draws per size")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--outdir", default=str(dc.REPO_ROOT / "runs"))
    args = ap.parse_args()

    if args.params is not None:
        params = json.loads(args.params)
    else:
        meta = Path(dc.REPO_ROOT / "runs/models/metadata.json")
        params = json.loads(meta.read_text())["params"] if meta.exists() else {}
    print(f"tree={args.model}  params={params}")

    tbl = dc.load_training_table()
    n_max = len(tbl.df)
    sizes = [s for s in (int(x) for x in args.sizes.split(",")) if s <= n_max]
    if n_max not in sizes:
        sizes.append(n_max)
    print(f"full table n={n_max}; sizes={sizes}")

    rng = np.random.default_rng(0)
    rows = []
    for n in sizes:
        maes = []
        for d in range(args.seeds if n < n_max else 1):
            idx = (np.arange(n_max) if n == n_max
                   else rng.choice(n_max, size=n, replace=False))
            delta, base = cv_subset(tbl, idx, args.model, params,
                                    args.folds, args.reps, 42 + d)
            maes.append((delta["MAE"], delta["R2"], base["MAE"]))
        maes = np.array(maes)
        rows.append(dict(n=n, mae=float(maes[:, 0].mean()),
                         mae_std=float(maes[:, 0].std()),
                         r2=float(maes[:, 1].mean()),
                         base_mae=float(maes[:, 2].mean())))
        print(f"  n={n:5d}  tree MAE={rows[-1]['mae']:.3f}±{rows[-1]['mae_std']:.3f}"
              f"  R²={rows[-1]['r2']:.3f}  (base {rows[-1]['base_mae']:.2f})")

    # GNN points recorded from GPU runs (pipeline/docs/gnn_delta.md learning-curve table).
    gnn = [(474, 3.34, 0.59), (1038, 3.49, 0.63), (1427, 3.40, 0.63)]

    out = Path(args.outdir)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    json.dump({"tree": rows, "gnn": [dict(n=n, mae=m, r2=r) for n, m, r in gnn],
               "model": args.model, "params": params},
              open(out / "data" / "learning_curve.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ns = [r["n"] for r in rows]
    mae = [r["mae"] for r in rows]
    std = [r["mae_std"] for r in rows]
    ax.errorbar(ns, mae, yerr=std, marker="o", capsize=3,
                label=f"tree ({args.model})", color="C0")
    ax.plot([g[0] for g in gnn], [g[1] for g in gnn], marker="s",
            ls="--", label="hybrid D-MPNN", color="C1")
    ax.set_xlabel("training-set size n (post-QC)")
    ax.set_ylabel("repeated-KFold CV MAE (kcal/mol)")
    ax.set_title("Benzoin ΔG Δ-learning — learning curve")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = out / "figs" / "learning_curve.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"\nSaved {p}")
    print(f"Saved {out / 'data' / 'learning_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
