#!/usr/bin/env python3
"""
Round-2 active-learning step 2 — score every round-2 row with a pair-grouped
bootstrap ENSEMBLE trained on round-1's 598-row table (same feature schema,
same xgb hyperparams as the shipped cross_delta_model.joblib), then rank
round-2 rows by ensemble prediction spread (std across bootstrap members) as
the uncertainty proxy. train_cross_delta.py's single "final" model has no
native uncertainty output (no quantile heads, unlike the homo ENSEMBLE72
champion) — bootstrapping the same xgb config over resampled pair_keys is the
standard substitute: high disagreement across resamples = the region of
descriptor space round-1's 299 pairs constrained the least, i.e. exactly
where a new DFT-SP label teaches the model the most.

Selection: rank by uncertainty (std) descending, take the top --n-select
rows, respecting one-row-per-unordered-pair (a pair's AB/BA duplicate rows
would double-count the same DFT-SP compute).

Usage
  python cross_benzoin/score_round2_active_learning.py --n-select 900 --n-boot 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402

TRAIN_TABLE = REPO / "data/cross_benzoin/cross_pilot_v1/cross_train_table.parquet"
ROUND2_FEATURES = REPO / "data/cross_benzoin/cross_round2/cross_round2_features.parquet"
FEATURE_LIST = REPO / "data/cross_benzoin/cross_pilot_v1/train_v1/models/feature_list.json"
OUT_CSV = REPO / "data/cross_benzoin/cross_round2/cross_round2_scored.csv"
OUT_SELECTED = REPO / "data/cross_benzoin/cross_round2/cross_round2_dft_selection.csv"
BASELINE_COL = "dG_gxtb_kcal"
TARGET_COL = "dG_orca_kcal"


def main() -> int:
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=40)
    ap.add_argument("--n-select", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feats = json.loads(FEATURE_LIST.read_text())
    train = pd.read_parquet(TRAIN_TABLE)
    round2 = pd.read_parquet(ROUND2_FEATURES)
    print(f"train table: {len(train)} rows, {train['pair_key'].nunique()} pairs")
    print(f"round2 features: {len(round2)} rows, {round2['pair_key'].nunique()} pairs")

    missing = [f for f in feats if f not in round2.columns]
    if missing:
        raise SystemExit(f"round2 feature table missing {len(missing)} columns the "
                          f"champion model needs, e.g. {missing[:5]} -- rerun assemble_cross_round2_features.py")

    Xtr_df = train[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xtr_df.median(numeric_only=True)
    Xtr = Xtr_df.fillna(medians)
    ytr = (train[TARGET_COL] - train[BASELINE_COL]).to_numpy()
    groups = train["pair_key"].to_numpy()
    uniq_pairs = np.unique(groups)

    Xte = round2[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)

    rng = np.random.default_rng(args.seed)
    preds = np.zeros((args.n_boot, len(round2)))
    for b in range(args.n_boot):
        boot_pairs = rng.choice(uniq_pairs, size=len(uniq_pairs), replace=True)
        idx = np.concatenate([np.where(groups == p)[0] for p in boot_pairs])
        m = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, args.seed + b)
        m.fit(Xtr.iloc[idx], ytr[idx])
        preds[b] = m.predict(Xte)
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot} done")

    correction_mean = preds.mean(axis=0)
    correction_std = preds.std(axis=0)
    dG_pred = round2[BASELINE_COL].to_numpy() + correction_mean

    out = round2[["id", "donor_id", "acceptor_id", "pair_key", "reaction_type",
                  "donor_smiles", "acceptor_smiles", "smiles",
                  "dG_xtb_kcal", BASELINE_COL]].copy()
    out["dG_pred_correction_mean"] = correction_mean
    out["dG_pred_correction_std"] = correction_std
    out["dG_pred"] = dG_pred
    out = out.sort_values("dG_pred_correction_std", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote scored table -> {OUT_CSV}")
    print(f"  uncertainty (std) distribution: min={correction_std.min():.3f} "
          f"median={np.median(correction_std):.3f} max={correction_std.max():.3f}")

    # One row per unordered pair for selection (keep the higher-uncertainty
    # orientation; DFT-SP will be run once per pair and applies to both rows).
    by_pair = out.loc[out.groupby("pair_key")["dG_pred_correction_std"].idxmax()]
    by_pair = by_pair.sort_values("dG_pred_correction_std", ascending=False).reset_index(drop=True)
    n_select_pairs = min(args.n_select, len(by_pair))
    selected = by_pair.head(n_select_pairs).copy()
    selected.to_csv(OUT_SELECTED, index=False)
    print(f"\nselected {len(selected)}/{len(by_pair)} unordered pairs for DFT-SP "
          f"(top uncertainty) -> {OUT_SELECTED}")
    print("  reaction_type distribution of selection:")
    print(selected["reaction_type"].value_counts().to_string())
    print("  reaction_type distribution of full round-2 pool (for comparison):")
    print(by_pair["reaction_type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
