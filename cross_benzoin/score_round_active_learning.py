#!/usr/bin/env python3
"""
Generalized version of score_round2_active_learning.py — scores a round's
unlabeled candidate pool with a pair-grouped bootstrap ENSEMBLE trained on a
given (already-labeled) training table, ranks by prediction-std across the
ensemble (epistemic-uncertainty proxy), and selects the top --n-select
unordered pairs for DFT-SP.

Round 3 uses this to score against the ROUND1+ROUND2 COMBINED table
(cross_train_table_combined.parquet, 2354 rows) rather than round-1 alone,
since that combined model is the best available reference at this point in
the active-learning loop.

Usage
  python cross_benzoin/score_round_active_learning.py --round 3 \
      --train-table data/cross_benzoin/cross_round2/cross_train_table_combined.parquet \
      --feature-list data/cross_benzoin/cross_round2/train_combined_v1/models/feature_list.json \
      --n-select 900 --n-boot 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402
from train_cross_delta import pair_split_labels  # noqa: E402
from train_cross_ensemble import _fit_ensemble  # noqa: E402

BASELINE_COL = "dG_gxtb_kcal"
TARGET_COL = "dG_orca_kcal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None,
                     help="round number for the standard cross_round{N} naming convention")
    ap.add_argument("--tag", default=None,
                     help="override rtag/rdir directly for non-round-numbered pools "
                          "(e.g. --tag screen10k -> data/cross_benzoin/screen10k/)")
    ap.add_argument("--candidates-path", type=Path, default=None,
                     help="override default {rtag}/{rtag}_features.parquet "
                          "(e.g. assemble_cross_round_features.py's --products-csv-suffixed output)")
    ap.add_argument("--train-table", type=Path, required=True)
    ap.add_argument("--feature-list", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=40)
    ap.add_argument("--n-select", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["xgb", "ensemble"], default="xgb",
                     help="'xgb' (default, unchanged from all prior rounds) bootstraps a single "
                          "depth-3 XGB per resample -- fast, the recipe every round through 6 used. "
                          "'ensemble' bootstraps the validated MLP+XGB stacking architecture "
                          "(train_cross_ensemble.py's MLPXGBEnsemble, the current best-validated "
                          "cross model as of round6, frozen MAE 2.582 vs the single-XGB champion's "
                          "3.132) per resample instead -- meaningfully slower (an MLP fit per "
                          "bootstrap draw, not just a tree ensemble) but a more accurate model "
                          "underlying the uncertainty estimate. Use this for the NEXT screening "
                          "campaign after round7 exhausts the current screen10k reservoir.")
    args = ap.parse_args()
    if args.round is None and args.tag is None:
        raise SystemExit("must pass --round N or --tag NAME")

    rtag = args.tag or f"cross_round{args.round}"
    rdir = REPO / "data/cross_benzoin" / rtag
    candidates_path = args.candidates_path or (rdir / f"{rtag}_features.parquet")
    out_csv = rdir / f"{rtag}_scored.csv"
    out_selected = rdir / f"{rtag}_dft_selection.csv"

    feats = json.loads(args.feature_list.read_text())
    train = pd.read_parquet(args.train_table)
    cand = pd.read_parquet(candidates_path)
    print(f"train table: {len(train)} rows, {train['pair_key'].nunique()} pairs ({args.train_table})")
    print(f"{rtag} candidate features: {len(cand)} rows, {cand['pair_key'].nunique()} pairs")

    # Never let the scoring ensemble train on candidates_v3's held-out
    # test/validation split -- a handful of rows in earlier rounds' custom
    # pair generators land there by chance (see
    # [[cross-round3-and-ensemble72-packaging-20260714]]); exclude them here
    # the same way train_cross_delta.py's final shipped-model fit does, so
    # the model that RANKS candidates for DFT-SP never touches held-out data
    # either.
    pair_split = pair_split_labels(train, verbose=False)
    if pair_split is not None:
        leak_mask = pair_split.isin(["test", "validation"]).to_numpy()
        if leak_mask.any():
            print(f"  excluding {leak_mask.sum()} train-table rows in candidates_v3's "
                  f"test/validation split from the scoring ensemble")
        train = train[~leak_mask].reset_index(drop=True)

    missing = [f for f in feats if f not in cand.columns]
    if missing:
        raise SystemExit(f"{rtag} feature table missing {len(missing)} columns the "
                          f"model needs, e.g. {missing[:5]} -- rerun assemble_cross_round_features.py")

    Xtr_df = train[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xtr_df.median(numeric_only=True)
    Xtr = Xtr_df.fillna(medians)
    ytr = (train[TARGET_COL] - train[BASELINE_COL]).to_numpy()
    groups = train["pair_key"].to_numpy()
    uniq_pairs = np.unique(groups)

    Xte = cand[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)

    rng = np.random.default_rng(args.seed)
    preds = np.zeros((args.n_boot, len(cand)))
    for b in range(args.n_boot):
        boot_pairs = rng.choice(uniq_pairs, size=len(uniq_pairs), replace=True)
        idx = np.concatenate([np.where(groups == p)[0] for p in boot_pairs])
        if args.model == "ensemble":
            sc, mlp, xa, xb = _fit_ensemble(Xtr.iloc[idx], ytr[idx], args.seed + b)
            Xte_s = sc.transform(Xte)
            preds[b] = (mlp.predict(Xte_s) + xa.predict(Xte) + xb.predict(Xte)) / 3.0
        else:
            m = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, args.seed + b)
            m.fit(Xtr.iloc[idx], ytr[idx])
            preds[b] = m.predict(Xte)
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot} done")

    correction_mean = preds.mean(axis=0)
    correction_std = preds.std(axis=0)
    dG_pred = cand[BASELINE_COL].to_numpy() + correction_mean

    out = cand[["id", "donor_id", "acceptor_id", "pair_key", "reaction_type",
               "donor_smiles", "acceptor_smiles", "smiles",
               "dG_xtb_kcal", BASELINE_COL]].copy()
    out["dG_pred_correction_mean"] = correction_mean
    out["dG_pred_correction_std"] = correction_std
    out["dG_pred"] = dG_pred
    out = out.sort_values("dG_pred_correction_std", ascending=False).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"\nwrote scored table -> {out_csv}")
    print(f"  uncertainty (std) distribution: min={correction_std.min():.3f} "
          f"median={np.median(correction_std):.3f} max={correction_std.max():.3f}")

    by_pair = out.loc[out.groupby("pair_key")["dG_pred_correction_std"].idxmax()]
    by_pair = by_pair.sort_values("dG_pred_correction_std", ascending=False).reset_index(drop=True)
    n_select_pairs = min(args.n_select, len(by_pair))
    selected = by_pair.head(n_select_pairs).copy()
    selected.to_csv(out_selected, index=False)
    print(f"\nselected {len(selected)}/{len(by_pair)} unordered pairs for DFT-SP "
          f"(top uncertainty) -> {out_selected}")
    print("  reaction_type distribution of selection:")
    print(selected["reaction_type"].value_counts().to_string())
    print("  reaction_type distribution of full candidate pool (for comparison):")
    print(by_pair["reaction_type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
