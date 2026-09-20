#!/usr/bin/env python
"""Robustness check on the Rec-2 AMBER verdict (2026-09-20, user-requested):
`homo_cross_joint_tabular_v2.py`'s single run found naive_merge beats
cross_only by 0.05 kcal on a 448-row holdout -- the script's own verdict
flagged that as "likely within noise at n=448" (AMBER, not GREEN). This
answers that directly, two ways, reusing the exact same data/features/model
recipe as the original script:

1. **Model-seed variance**: refit cross_only / naive_merge / naive_merge_weighted
   over N_SEEDS different XGBoost random_state values (colsample_bytree=0.7
   means this actually changes the fitted trees, not a no-op) -- if the
   ranking (naive_merge < cross_only) holds across seeds, that's evidence
   it's not a lucky single fit.
2. **Holdout bootstrap**: resample the 448-row test set with replacement
   B=10,000 times using the (fixed) seed-0 predictions -- if the sign of
   (MAE_cross_only - MAE_naive_merge) is positive in the vast majority of
   resamples, the win isn't an artifact of which 448 pairs happen to be in
   the holdout.

Usage:
    /home/schen3/venv/nequip/bin/python cross_benzoin/homo_cross_joint_tabular_v2_robustness.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from homo_cross_joint_tabular_v2 import load_cross, load_homo, FEATURE_LIST  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HOMO_TABLE = REPO / "data/cross_benzoin/homo_standalone/homo_standalone_full_library_table_slim260.parquet"
OUT = REPO / "data/analysis/homo_cross_gap/homo_cross_joint_tabular_v2_robustness.json"
N_SEEDS = 5
N_BOOTSTRAP = 10000


def _xgb(seed):
    return XGBRegressor(n_estimators=1800, max_depth=9, learning_rate=0.02, subsample=0.75,
                         colsample_bytree=0.7, min_child_weight=5, n_jobs=16, eval_metric="mae",
                         random_state=seed)


def _predict(model, te, feats):
    F = feats + ["is_homo"]
    dp = model.predict(te[F].to_numpy())
    return te["dG_b973c_kcal"].to_numpy() + dp


def main() -> int:
    t0 = time.time()
    feats = json.loads(FEATURE_LIST.read_text())
    cross = load_cross(feats)
    homo = load_homo(HOMO_TABLE, feats)
    xtr = cross[cross.new_scaffold_split == "train"].reset_index(drop=True)
    xte = cross[cross.new_scaffold_split == "test"].reset_index(drop=True)
    yte_true = xte["dG_r2scan_kcal"].to_numpy()
    F = feats + ["is_homo"]

    both = pd.concat([homo[F + ["delta"]], xtr[F + ["delta"]]], ignore_index=True)
    w_homo = len(xtr) / max(len(homo), 1)
    weights = np.concatenate([np.full(len(homo), w_homo), np.ones(len(xtr))])

    print(f"homo {len(homo)}  cross clean-train {len(xtr)}  cross holdout {len(xte)}", flush=True)

    per_seed = {"cross_only": [], "naive_merge": [], "naive_merge_weighted": []}
    seed0_preds = {}
    for seed in range(N_SEEDS):
        m = _xgb(seed); m.fit(xtr[F].to_numpy(), xtr["delta"].to_numpy())
        p = _predict(m, xte, feats)
        mae = mean_absolute_error(yte_true, p)
        per_seed["cross_only"].append(mae)
        if seed == 0:
            seed0_preds["cross_only"] = p

        m = _xgb(seed); m.fit(both[F].to_numpy(), both["delta"].to_numpy())
        p = _predict(m, xte, feats)
        mae = mean_absolute_error(yte_true, p)
        per_seed["naive_merge"].append(mae)
        if seed == 0:
            seed0_preds["naive_merge"] = p

        m = _xgb(seed); m.fit(both[F].to_numpy(), both["delta"].to_numpy(), sample_weight=weights)
        p = _predict(m, xte, feats)
        mae = mean_absolute_error(yte_true, p)
        per_seed["naive_merge_weighted"].append(mae)
        if seed == 0:
            seed0_preds["naive_merge_weighted"] = p

        print(f"seed {seed}: cross_only={per_seed['cross_only'][-1]:.4f}  "
              f"naive_merge={per_seed['naive_merge'][-1]:.4f}  "
              f"naive_merge_weighted={per_seed['naive_merge_weighted'][-1]:.4f}  "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

    seed_summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                    for k, v in per_seed.items()}
    gap_per_seed = [c - m for c, m in zip(per_seed["cross_only"], per_seed["naive_merge"])]
    seed_summary["gap_cross_only_minus_naive_merge"] = {
        "mean": float(np.mean(gap_per_seed)), "std": float(np.std(gap_per_seed)),
        "values": gap_per_seed, "n_seeds_favoring_merge": int(sum(g > 0 for g in gap_per_seed)),
    }
    print(f"\n=== across {N_SEEDS} seeds ===")
    for k in ("cross_only", "naive_merge", "naive_merge_weighted"):
        print(f"{k}: {seed_summary[k]['mean']:.4f} +/- {seed_summary[k]['std']:.4f}")
    print(f"gap (cross_only - naive_merge): {seed_summary['gap_cross_only_minus_naive_merge']['mean']:+.4f} "
          f"+/- {seed_summary['gap_cross_only_minus_naive_merge']['std']:.4f}, "
          f"{seed_summary['gap_cross_only_minus_naive_merge']['n_seeds_favoring_merge']}/{N_SEEDS} seeds favor merge")

    rng = np.random.default_rng(0)
    n = len(yte_true)
    boot_gaps = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        mae_co = mean_absolute_error(yte_true[idx], seed0_preds["cross_only"][idx])
        mae_nm = mean_absolute_error(yte_true[idx], seed0_preds["naive_merge"][idx])
        boot_gaps[b] = mae_co - mae_nm
    ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5])
    frac_favoring_merge = float(np.mean(boot_gaps > 0))
    bootstrap_summary = {
        "n_bootstrap": N_BOOTSTRAP, "n_holdout": n,
        "mean_gap": float(boot_gaps.mean()), "std_gap": float(boot_gaps.std()),
        "ci_95_lo": float(ci_lo), "ci_95_hi": float(ci_hi),
        "frac_resamples_favoring_merge": frac_favoring_merge,
    }
    print(f"\n=== holdout bootstrap ({N_BOOTSTRAP} resamples of the fixed seed-0 predictions) ===")
    print(f"gap (cross_only - naive_merge): mean={bootstrap_summary['mean_gap']:+.4f}, "
          f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"fraction of resamples where naive_merge wins: {frac_favoring_merge:.1%}")
    if ci_lo > 0:
        verdict = "GREEN: 95% CI for the gap excludes zero -- naive_merge's win is not holdout-sampling noise."
    elif frac_favoring_merge > 0.90:
        verdict = "AMBER-GREEN: CI touches zero but >90% of resamples favor naive_merge -- probably real, not proven."
    else:
        verdict = "AMBER/RED: CI straddles zero substantially -- consistent with the original 'likely noise' call."
    print("\n" + verdict)

    out = {
        "built_by": "cross_benzoin/homo_cross_joint_tabular_v2_robustness.py",
        "n_seeds": N_SEEDS, "seed_summary": seed_summary,
        "bootstrap": bootstrap_summary, "verdict": verdict,
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n-> {OUT} ({out['runtime_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
