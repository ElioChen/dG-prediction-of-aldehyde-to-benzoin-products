#!/usr/bin/env python3
"""
(a) Decide whether the CHO-environment category helps as a MODEL FEATURE, and
build the per-category APPLICABILITY-DOMAIN table that should ship with the model.

  1. Repeated-KFold CV MAE with the 63 descriptors vs +3 category one-hots
     (cho_category.FEATURE_COLS). Tells us if the explicit category sharpens the
     model or is redundant with the existing aromaticity descriptors.
  2. Per-category OOF MAE/RMSE/n from the production (63-feature) model → the
     applicability-domain confidence table (e.g. carbo-aromatic ±2.4, hetero ±3.4).

Writes runs/data/category_ad.json (consumed by assemble_model / metadata).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

import delta_core as dc
from cho_category import category_features

SEED = 42


def oof(X, y, params, folds=5, repeats=3):
    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    s = np.zeros(len(X)); c = np.zeros(len(X))
    for tr, te in rkf.split(X):
        m = dc.build_model("xgb", params, SEED); m.fit(X[tr], y[tr])
        s[te] += m.predict(X[te]); c[te] += 1
    return s / np.maximum(c, 1)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=dc.DEFAULT_FEATURIZE_PARQUET)
    args = ap.parse_args()
    tbl = dc.load_training_table(parquet=args.parquet)
    # Use the shipped model's tuned params so the AD matches production; fall back
    # to default xgb (fine for the relative feature comparison).
    meta = dc.REPO_ROOT / "runs/models/metadata.json"
    params = json.loads(meta.read_text()).get("params", {}) if meta.exists() else {}
    print(f"params: {'tuned (runs/models)' if params else 'default xgb'}")
    catdf, cls = category_features(tbl.df["SMILES"])
    cls = np.array(cls)

    X63 = tbl.X.to_numpy()
    X66 = np.hstack([X63, catdf.to_numpy()])
    print(f"n={len(tbl.df)}  category counts:",
          {c: int((cls == c).sum()) for c in sorted(set(cls))})

    out = {}
    for tag, X in [("desc63", X63), ("desc63+cat", X66)]:
        pred = tbl.dG_xtb + oof(X, tbl.y, params)
        out[tag] = dict(MAE=float(mean_absolute_error(tbl.dG_dft, pred)),
                        RMSE=float(np.sqrt(mean_squared_error(tbl.dG_dft, pred))),
                        R2=float(r2_score(tbl.dG_dft, pred)))
        print(f"  {tag:12s} MAE={out[tag]['MAE']:.3f}  RMSE={out[tag]['RMSE']:.3f}"
              f"  R²={out[tag]['R2']:.3f}")
    delta = out["desc63"]["MAE"] - out["desc63+cat"]["MAE"]
    print(f"  category feature ΔMAE = {delta:+.3f} "
          f"({'helps' if delta > 0.02 else 'negligible' if abs(delta) <= 0.02 else 'hurts'})")

    # per-category applicability domain from the production (63-feature) OOF
    pred63 = tbl.dG_xtb + oof(X63, tbl.y, params)
    ae = np.abs(tbl.dG_dft - pred63)
    ad = {}
    print("\nPer-category applicability domain (63-feature model):")
    print("  category           n    MAE   RMSE   dG_med")
    for c in sorted(set(cls)):
        sel = cls == c
        if sel.sum() == 0:
            continue
        ad[c] = dict(n=int(sel.sum()), MAE=float(ae[sel].mean()),
                     RMSE=float(np.sqrt((ae[sel] ** 2).mean())),
                     dG_median=float(np.median(tbl.dG_dft[sel])))
        print(f"  {c:16s} {ad[c]['n']:4d}  {ad[c]['MAE']:.2f}  {ad[c]['RMSE']:.2f}"
              f"  {ad[c]['dG_median']:6.2f}")

    res = dict(n=len(tbl.df), feature_comparison=out,
               category_feature_delta_mae=delta, applicability_domain=ad)
    outp = dc.REPO_ROOT / "runs/data/category_ad.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(outp, "w"), indent=2)
    print(f"\nSaved {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
