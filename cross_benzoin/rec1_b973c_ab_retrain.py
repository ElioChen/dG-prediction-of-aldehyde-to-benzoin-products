#!/usr/bin/env python
"""Rec-1 GREEN-branch paired A/B proxy.

On the SAME regenerated funnel_v3-geometry slice (holdout grp=='holdout' vs the rest),
retrain the tabular Delta-model with two baselines and read the holdout-MAE gap:

    y_A = dG_r2scan_regen - dG_gxtb_regen      (current production baseline)
    y_B = dG_r2scan_regen - dG_b973c_regen     (candidate B97-3c baseline)

r2SCAN target and baseline are BOTH the freshly regenerated values on the identical
geometry, so the target is conformer-self-consistent (the stored dG_orca label is on a
different, purged geometry -- repro_r2scan std ~4.5 -- and using it would inject that
mismatch into y and mask the lever). Absolute MAE is inflated by the small train slice;
the A-vs-B gap on identical data/target is the signal.

  python rec1_b973c_ab_retrain.py \
      --merged data/cross_benzoin/rec1_prodgeom_recheck/rec1_b973c_intermediate_1999_merged.csv \
      --out    data/cross_benzoin/rec1_prodgeom_recheck/ab_retrain_result.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "cross_benzoin"))
import delta_core as dc  # noqa: E402

FEATS = json.loads((REPO / "data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json").read_text())
TABLE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
XGB_HP = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}


def _fit_eval(tr, te, feats, base_col, tgt_col, seed=42):
    med = tr[feats].apply(pd.to_numeric, errors="coerce").median()
    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").fillna(med).to_numpy()
    Xte = te[feats].apply(pd.to_numeric, errors="coerce").fillna(med).to_numpy()
    ytr = (tr[tgt_col] - tr[base_col]).to_numpy()
    m = dc.build_model("xgb", XGB_HP, seed)
    m.fit(Xtr, ytr)
    pred = te[base_col].to_numpy() + m.predict(Xte)
    truth = te[tgt_col].to_numpy()
    delta = dc.metrics_vs_dft(truth, pred)
    base = dc.metrics_vs_dft(truth, te[base_col].to_numpy())
    return delta, base, np.abs(truth - pred)


def _bootstrap_gap(ae_a, ae_b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ae_a))
    d = [ae_a[i].mean() - ae_b[i].mean() for i in (rng.choice(idx, len(idx), replace=True) for _ in range(n))]
    d = np.sort(d)
    return float(np.mean(d)), float(d[int(0.05 * n)]), float(d[int(0.95 * n)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    mg = pd.read_csv(a.merged)
    mg = mg[mg["error"].isna() & mg["dG_b973c_kcal"].notna() & mg["dG_r2scan_kcal"].notna()].copy()
    tab = pd.read_parquet(TABLE, columns=["id"] + FEATS + ["dG_gxtb_kcal", "dG_orca_kcal"])
    tab = tab.rename(columns={"dG_gxtb_kcal": "dG_gxtb_stored", "dG_orca_kcal": "dG_orca_stored"})
    df = mg.merge(tab, left_on="pid", right_on="id", how="inner")
    print(f"{len(mg)} usable regen rows -> {len(df)} joined to features")

    te = df[df["grp"] == "holdout"].reset_index(drop=True)
    tr = df[df["grp"] != "holdout"].reset_index(drop=True)
    print(f"train {len(tr)}  holdout {len(te)}")
    if len(te) < 50 or len(tr) < 200:
        print("insufficient rows for a meaningful A/B"); return 1

    out = {"n_train": len(tr), "n_holdout": len(te), "seeds": a.seeds,
           "self_consistent_target": {}, "stored_label_target": {}}
    for tag, tgt in [("self_consistent_target", "dG_r2scan_kcal"),
                     ("stored_label_target", "dG_orca_stored")]:
        block = {}
        ae = {}
        for name, base in [("gxtb", "dG_gxtb_kcal"), ("b973c", "dG_b973c_kcal")]:
            if tag == "stored_label_target" and name == "gxtb":
                base = "dG_gxtb_stored"   # production apples-to-apples for the stored target
            maes, bmaes, aes = [], [], []
            for s in range(a.seeds):
                d, b, aev = _fit_eval(tr, te, FEATS, base, tgt, seed=42 + s)
                maes.append(d["MAE"]); bmaes.append(b["MAE"]); aes.append(aev)
            ae[name] = np.mean(aes, axis=0)
            block[name] = {"delta_MAE": round(float(np.mean(maes)), 3),
                           "delta_MAE_sd": round(float(np.std(maes)), 3),
                           "raw_baseline_MAE": round(float(np.mean(bmaes)), 3)}
        gap_mean, gap_lo, gap_hi = _bootstrap_gap(ae["gxtb"], ae["b973c"])
        block["gap_gxtb_minus_b973c"] = {"mean": round(gap_mean, 3),
                                         "ci90": [round(gap_lo, 3), round(gap_hi, 3)],
                                         "b973c_better": gap_lo > 0}
        out[tag] = block
        print(f"\n=== target = {tag} ===")
        for k in ("gxtb", "b973c"):
            print(f"  {k:6s} baseline: raw MAE {block[k]['raw_baseline_MAE']:.3f}  "
                  f"-> Delta-model holdout MAE {block[k]['delta_MAE']:.3f} (+/-{block[k]['delta_MAE_sd']:.3f})")
        print(f"  gap (gxtb - b973c) = {block['gap_gxtb_minus_b973c']['mean']:+.3f}  "
              f"90% CI {block['gap_gxtb_minus_b973c']['ci90']}  "
              f"b973c clearly better: {block['gap_gxtb_minus_b973c']['b973c_better']}")

    sc = out["self_consistent_target"]["gap_gxtb_minus_b973c"]
    if sc["b973c_better"] and sc["mean"] > 0.5:
        v = (f"A/B PROXY POSITIVE: B97-3c baseline Delta-model beats g-xTB baseline by "
             f"{sc['mean']:.2f} kcal (90% CI {sc['ci90']}) on the self-consistent target. "
             f"The std-ratio translates to dG-level accuracy -> pitch the full ~23k relabel campaign.")
    elif sc["b973c_better"]:
        v = (f"A/B PROXY WEAK-POSITIVE: b973c better by {sc['mean']:.2f} kcal but < 0.5; "
             f"marginal -- weigh full-campaign cost carefully.")
    else:
        v = (f"A/B PROXY NULL: no clear B97-3c advantage after ML correction "
             f"(gap {sc['mean']:+.2f}, CI {sc['ci90']}). The std-ratio was constant-offset "
             f"absorption, not accuracy. Do NOT pitch the full campaign; write up + move to Rec 2.")
    out["verdict"] = v
    print(f"\n>>> {v}")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
