#!/usr/bin/env python
"""Cheap-CPU GBM bake-off on the H-SPOC local-3D descriptor set, scaffold-disjoint.

STATUS.md ranks "H-SPOC local descriptors + XGB" as the best zero-extra-compute
baseline (2.42 / 4.05 MAE tuned). XGB there was the only GBM ever tried on this feature
set. This screens alternative gradient-boosted / tree heads on the SAME features, split
and QC, so a full-scale rerun after the descriptor-library rebuild only has to carry the
head that actually wins:

  xgb   - XGBRegressor (the incumbent, tuned-ish params from tune_hspoc_xgb DEFAULT)
  hgb   - sklearn HistGradientBoostingRegressor (LightGBM-style histogram boosting)
  rf    - RandomForestRegressor
  et    - ExtraTreesRegressor

All available in the nhc-workflow venv (no installs). Runs on whatever
{which}_all.csv currently holds -- pre-array that is the partial ~42k aldehyde set, so
treat absolute numbers as a same-scale screen, not the honest champion-scale figure.

Usage:
  PY=/home/schen3/venv/nhc-workflow/bin/python
  $PY pipeline/bde/gbm_bakeoff_hspoc.py --which aldehydes \
     --split-file data/cross_benzoin/homo_v6/aldehydes_scaffold_split_from_dG.csv \
     --out runs/logs/scaffold_disjoint_bde/gbm_bakeoff_aldehydes.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc import norm_id, qc_filter             # noqa: E402
from train_local3d_baseline import LOCAL_FEATURES  # noqa: E402

_repo_h = Path(__file__).resolve().parents[2] / "data/cross_benzoin/homo_v6"
H = (Path(os.environ["BDE_HOMO_V6"]) if os.environ.get("BDE_HOMO_V6")
     else _repo_h if _repo_h.exists()
     else Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6"))

XGB_PARAMS = dict(n_estimators=600, max_depth=4, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                  n_jobs=8, random_state=0)


def build_models():
    m = {
        "hgb": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             early_stopping=True, random_state=0),
        "rf": RandomForestRegressor(n_estimators=500, n_jobs=8, random_state=0),
        "et": ExtraTreesRegressor(n_estimators=500, n_jobs=8, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        m = {"xgb": XGBRegressor(**XGB_PARAMS), **m}
    except ImportError:
        print("WARN: xgboost not importable in this env, skipping xgb head")
    return m


def score(y, p):
    return dict(mae=float(mean_absolute_error(y, p)),
               rmse=float(root_mean_squared_error(y, p)),
               r2=float(r2_score(y, p)),
               spearman=float(spearmanr(y, p).statistic))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--target", choices=["bde", "bdfe"], default="bde")
    ap.add_argument("--split-file", type=Path, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    feats = LOCAL_FEATURES[args.which]
    mol = pd.read_csv(H / f"{args.which}_all.csv", dtype={"id": str},
                      keep_default_na=False, low_memory=False)
    if "error" in mol.columns:
        mol = mol[mol["error"] == ""]
    for c in feats:
        mol[c] = pd.to_numeric(mol.get(c), errors="coerce")
    mol["id"] = norm_id(mol["id"])
    mol = mol[["id"] + [c for c in feats if c in mol.columns]].drop_duplicates("id")

    labels = pd.read_csv(H / f"{args.which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    labels["id"] = norm_id(labels["id"])
    ycol = f"{args.target}_gxtb_kcal"
    labels = labels.dropna(subset=[ycol]).drop_duplicates("id")
    labels = labels[qc_filter(labels[ycol])]

    df = labels.merge(mol, on="id", how="inner")
    use_feats = [c for c in feats if c in df.columns]
    df = df.dropna(subset=use_feats, how="all").reset_index(drop=True)
    df[use_feats] = df[use_feats].fillna(df[use_feats].median())
    # drop zero-variance columns (constant in this subset -> HistGBM binning crashes)
    dropped = [c for c in use_feats if df[c].nunique(dropna=False) <= 1]
    if dropped:
        print(f"  dropping {len(dropped)} zero-variance features: {dropped}")
        use_feats = [c for c in use_feats if c not in dropped]

    sf = pd.read_csv(args.split_file, dtype={"id": str})[["id", "scaffold_split"]]
    sf["id"] = norm_id(sf["id"])
    df = df.merge(sf, on="id", how="inner")
    df["scaffold_split"] = df["scaffold_split"].replace({"validation": "train"})
    tr = df[df["scaffold_split"] == "train"]
    te = df[df["scaffold_split"] == "test"]
    print(f"{args.which}: {len(df)} labelled+featurized rows | "
          f"train {len(tr)} / test {len(te)} | {len(use_feats)} features", flush=True)

    Xtr, ytr = tr[use_feats].to_numpy(), tr[ycol].to_numpy()
    Xte, yte = te[use_feats].to_numpy(), te[ycol].to_numpy()

    results = {}
    for name, model in build_models().items():
        t0 = time.time()
        try:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            results[name] = {**score(yte, pred), "fit_sec": round(time.time() - t0, 1)}
            r = results[name]
            print(f"  {name:4s}  MAE {r['mae']:.3f}  RMSE {r['rmse']:.3f}  "
                  f"R2 {r['r2']:.3f}  rho {r['spearman']:.3f}  ({r['fit_sec']}s)", flush=True)
        except Exception as e:
            results[name] = {"error": repr(e)}
            print(f"  {name:4s}  FAILED: {e!r}", flush=True)

    ok = {k: v for k, v in results.items() if "mae" in v}
    best = min(ok, key=lambda k: ok[k]["mae"]) if ok else None
    out = dict(which=args.which, target=args.target, n_train=len(tr), n_test=len(te),
               n_features=len(use_feats), split_file=str(args.split_file),
               data_note="partial aldehydes_all.csv if run pre-featurize-array 26316404",
               results=results, best_by_mae=best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"best by MAE: {best}  ->  wrote {args.out}")


if __name__ == "__main__":
    main()
