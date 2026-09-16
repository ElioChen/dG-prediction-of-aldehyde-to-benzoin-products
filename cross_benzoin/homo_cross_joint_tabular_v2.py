#!/usr/bin/env python
"""Rec-2 re-test at 257-feat + B97-3c-baseline scale (PROJECT_PLAN.md 2.10.c,
REC2_PREP_PLAN_20260908.md step 6): does homo_cross_joint_tabular.py's
naive_merge +0.11 kcal survive now that (a) the champion schema is the full
257-feature space instead of a 72-feat shared proxy, (b) the baseline is
B97-3c not g-xTB (residual floor 0.916 vs 3.785 -- FINDING.md's Task C ran
before this breakthrough), and (c) homo is closer to its full ~184k scale
instead of the balanced-by-luck 30k homo_unify_v1 subsample -- FINDING.md's
own caveat was that at full scale (~219k homo : 35k cross =~ 6:1) naive
per-row pooling would dilute the cross signal unless homo is down-weighted.

Three conditions beyond the Task C baseline pair:
  cross_only            XGB on cross clean-train only (unchanged baseline)
  naive_merge            + homo, uniform per-row weight (repeats the Task C
                          recipe verbatim at the new scale -- expected to
                          UNDER-perform cross_only if the dilution warning is
                          real, since homo will now outnumber cross rows)
  naive_merge_weighted   + homo, sample_weight so sum(homo weight) ==
                          n_cross_train (restores the ~1:1 *effective*
                          balance that produced the original +0.11, per
                          FINDING.md's explicit fix)

Inputs (both already built to the SAME 257-feature champion schema by
construction -- assemble_homo_standalone_table.py's ALDEHYDE_FEATS/
PRODUCT_FEATS/_rdkit_block/interaction_* helpers are the exact functions
the cross Tier B table was built from, so no re-alignment/renaming of the
feature columns themselves is needed, only the label/baseline column names,
which differ by convention between the two tables):
  cross  data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet
         label=dG_r2scan_kcal baseline=dG_b973c_kcal split=new_scaffold_split
  homo   --homo-table (assemble_homo_standalone_table.py --full-library output)
         label=dG_orca_kcal (renamed from dG_r2scan_kcal by that script)
         baseline=dG_b973c_kcal split=new_scaffold_split

Eval strictly on the cross scaffold-disjoint TEST split (n=448, frozen since
round 7) -- homo rows only ever enter the train pool, matching Task C and
keeping the comparison apples-to-apples with the deployed champion's own
holdout.

  /home/schen3/venv/nhc-workflow/bin/python cross_benzoin/homo_cross_joint_tabular_v2.py \
      --homo-table data/cross_benzoin/homo_standalone/homo_standalone_full_library.parquet
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

REPO = Path(__file__).resolve().parents[1]
CROSS_TABLE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet"
FEATURE_LIST = REPO / "data/cross_benzoin/feature_list_257_no_nCHO_v2.json"
OUT = REPO / "data/analysis/homo_cross_gap"
OUT.mkdir(parents=True, exist_ok=True)

# CHAMPION.md single-XGB Delta-model reference points, same table, same holdout
CROSS_REF = {"ensemble_only_mae": 0.603, "blend_mae": 0.528, "single_xgb_mae": 0.632,
             "b973c_baseline_mae": None}  # baseline MAE not yet in CHAMPION.md as a standalone number


def _xgb():
    return XGBRegressor(n_estimators=1800, max_depth=9, learning_rate=0.02, subsample=0.75,
                         colsample_bytree=0.7, min_child_weight=5, n_jobs=16, eval_metric="mae")


def load_cross(feats: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(CROSS_TABLE)
    df["delta"] = df["dG_r2scan_kcal"] - df["dG_b973c_kcal"]
    df["is_homo"] = 0
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"cross table missing {len(missing)} champion feature cols, e.g. {missing[:5]}")
    keep = ["id", "new_scaffold_split", "delta", "dG_r2scan_kcal", "dG_b973c_kcal", "is_homo"] + feats
    return df[keep].copy()


def load_homo(path: Path, feats: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    label_col = "dG_orca_kcal" if "dG_orca_kcal" in df.columns else "dG_r2scan_kcal"
    df["delta"] = df[label_col] - df["dG_b973c_kcal"]
    df["is_homo"] = 1
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"homo table missing {len(missing)} champion feature cols, e.g. {missing[:5]}")
    df = df.dropna(subset=["delta"] + feats).reset_index(drop=True)
    keep = ["id", "new_scaffold_split", "delta", label_col, "dG_b973c_kcal", "is_homo"] + feats
    return df[keep].rename(columns={label_col: "dG_r2scan_kcal"}).copy()


def _ev(model, te, feats):
    F = feats + ["is_homo"]
    dp = model.predict(te[F].to_numpy())
    yp = te["dG_b973c_kcal"].to_numpy() + dp
    yt = te["dG_r2scan_kcal"].to_numpy()
    return {"mae": float(mean_absolute_error(yt, yp)), "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--homo-table", type=Path, required=True,
                     help="assemble_homo_standalone_table.py --full-library output (.parquet)")
    ap.add_argument("--provisional", action="store_true",
                     help="label output as provisional (homo labels still draining)")
    args = ap.parse_args()

    t0 = time.time()
    feats = json.loads(FEATURE_LIST.read_text())
    cross = load_cross(feats)
    homo = load_homo(args.homo_table, feats)
    xtr = cross[cross.new_scaffold_split == "train"].reset_index(drop=True)
    xte = cross[cross.new_scaffold_split == "test"].reset_index(drop=True)
    print(f"homo {len(homo)}  cross clean-train {len(xtr)}  cross holdout {len(xte)}  "
          f"homo:cross ratio {len(homo) / max(len(xtr), 1):.2f}:1", flush=True)
    F = feats + ["is_homo"]

    res = {}

    m = _xgb(); m.fit(xtr[F].to_numpy(), xtr["delta"].to_numpy())
    res["cross_only"] = _ev(m, xte, feats)
    print("cross_only", res["cross_only"], flush=True)

    both = pd.concat([homo[F + ["delta"]], xtr[F + ["delta"]]], ignore_index=True)
    m = _xgb(); m.fit(both[F].to_numpy(), both["delta"].to_numpy())
    res["naive_merge"] = _ev(m, xte, feats)
    print("naive_merge", res["naive_merge"], flush=True)

    w_homo = len(xtr) / max(len(homo), 1)
    weights = np.concatenate([np.full(len(homo), w_homo), np.ones(len(xtr))])
    m = _xgb(); m.fit(both[F].to_numpy(), both["delta"].to_numpy(), sample_weight=weights)
    res["naive_merge_weighted"] = _ev(m, xte, feats)
    print(f"naive_merge_weighted (w_homo={w_homo:.4f})", res["naive_merge_weighted"], flush=True)

    m = _xgb(); m.fit(homo[F].to_numpy(), homo["delta"].to_numpy())
    res["homo_only_zeroshot"] = _ev(m, xte, feats)
    m2 = _xgb(); m2.fit(xtr[F].to_numpy(), xtr["delta"].to_numpy(), xgb_model=m.get_booster())
    res["finetune"] = _ev(m2, xte, feats)
    print("homo_only_zeroshot", res["homo_only_zeroshot"], flush=True)
    print("finetune", res["finetune"], flush=True)

    base = res["cross_only"]["mae"]
    best_transfer = min(res["naive_merge"]["mae"], res["naive_merge_weighted"]["mae"], res["finetune"]["mae"])
    gain = base - best_transfer
    if gain > 0.10:
        verdict = (f"GREEN: homo transfer helps ({gain:+.3f} kcal over cross_only 257-feat "
                   f"b973c-baseline XGB {base:.3f}).")
    elif gain > 0.03:
        verdict = f"AMBER: marginal homo-transfer gain ({gain:+.3f} kcal), likely within noise at n=448."
    else:
        verdict = f"RED: no homo-transfer gain at this scale ({gain:+.3f} kcal). Rec-2's -0.11 did not survive."
    out = {"built_by": "cross_benzoin/homo_cross_joint_tabular_v2.py",
           "provisional": bool(args.provisional),
           "homo_table": str(args.homo_table),
           "n_homo": len(homo), "n_cross_train": len(xtr), "n_cross_holdout": len(xte),
           "homo_cross_ratio": len(homo) / max(len(xtr), 1),
           "w_homo_per_row": w_homo,
           "features": f"{len(feats)} champion schema (v2, no n_CHO) + is_homo",
           "results": res, "cross_reference_full_champion": CROSS_REF,
           "verdict": verdict, "runtime_min": round((time.time() - t0) / 60, 1)}
    tag = "_provisional" if args.provisional else ""
    outfile = OUT / f"homo_cross_joint_tabular_v2{tag}.json"
    outfile.write_text(json.dumps(out, indent=2, default=float))
    print("\n" + verdict)
    print(f"-> {outfile}  ({out['runtime_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
