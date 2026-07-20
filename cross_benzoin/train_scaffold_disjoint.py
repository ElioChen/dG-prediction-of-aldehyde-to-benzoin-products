#!/usr/bin/env python3
"""
Train + evaluate the cross-benzoin champion (single-XGB) and MLP+XGB
ensemble on the NEW scaffold-disjoint split (new_scaffold_split column,
from rebuild_scaffold_disjoint_split.py + the pair-level re-derivation),
replacing the old candidates_v3 molecule-level split found to leak scaffolds
severely (see memory cross-five-diagnostics-20260717.md).

Unlike train_cross_delta.py/train_cross_ensemble.py's frozen_holdout_eval
(hardcoded to candidates_v3's own pair_split_labels()), this script takes
the pre-computed `new_scaffold_split` column directly -- train=fit rows,
test=honest held-out eval rows (450, ~15x the old n=29), validation+mixed
rows excluded from both (mixed = donor/acceptor scaffolds landed in
different splits, ambiguous/leaky if used either way).

Usage:
  python cross_benzoin/train_scaffold_disjoint.py \
      --table data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_split_labeled.parquet \
      --outdir data/cross_benzoin/cross_round7/scaffold_disjoint_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402
from train_cross_delta import _feature_blocks, TARGET_COL, BASELINE_COL  # noqa: E402
from train_cross_ensemble import MLPXGBEnsemble, _fit_ensemble  # noqa: E402


def repeated_group_cv_xgb(df, feats, groups, folds, repeats, seed):
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xdf.median(numeric_only=True)
    X = Xdf.fillna(medians)
    y = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()
    oof_sum = np.zeros(len(df)); oof_cnt = np.zeros(len(df))
    for r in range(repeats):
        gkf = GroupKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        for tr, te in gkf.split(X, y, groups=groups):
            m = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, seed)
            m.fit(X.iloc[tr], y[tr])
            oof_sum[te] += m.predict(X.iloc[te]); oof_cnt[te] += 1
    oof = oof_sum / np.maximum(oof_cnt, 1)
    pred_dft = df[BASELINE_COL].to_numpy() + oof
    return dc.metrics_vs_dft(df[TARGET_COL].to_numpy(), pred_dft)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    (args.outdir / "models").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.table)
    train_df = df[df["new_scaffold_split"] == "train"].reset_index(drop=True)
    test_df = df[df["new_scaffold_split"] == "test"].reset_index(drop=True)
    print(f"loaded {len(df)} total rows -> clean train={len(train_df)}, clean test={len(test_df)} "
          f"(excluded: validation={len(df[df['new_scaffold_split']=='validation'])}, "
          f"mixed={len(df[df['new_scaffold_split']=='mixed'])})")

    all_feats = [c for c in df.columns if c not in {
        "id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
        "donor_smiles", "acceptor_smiles", "smiles", "dG_xtb_kcal", "dG_gxtb_kcal",
        "dG_orca_kcal", "donor_scaf_split", "acceptor_scaf_split", "new_scaffold_split"}]
    feats = _feature_blocks(all_feats)["all_raw_blocks+mordred"]
    print(f"{len(feats)} features (all_raw_blocks+mordred, matches production champion)")

    # --- CV on clean-train pool only (stability estimate, not the headline number) ---
    groups = train_df["pair_key"].to_numpy()
    cv_xgb = repeated_group_cv_xgb(train_df, feats, groups, args.folds, args.repeats, args.seed)
    print(f"\n[clean-train CV, single-XGB] MAE={cv_xgb['MAE']:.3f} R2={cv_xgb['R2']:.3f}")

    # --- held-out test: single-XGB champion, fit on ALL clean train, eval on clean test ---
    medians = train_df[feats].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    Xtr = train_df[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)
    Xte = test_df[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)
    ytr = (train_df[TARGET_COL] - train_df[BASELINE_COL]).to_numpy()

    m_xgb = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, args.seed)
    m_xgb.fit(Xtr, ytr)
    pred_test_xgb = test_df[BASELINE_COL].to_numpy() + m_xgb.predict(Xte)
    holdout_xgb = dc.metrics_vs_dft(test_df[TARGET_COL].to_numpy(), pred_test_xgb)
    base_gxtb_holdout = dc.metrics_vs_dft(test_df[TARGET_COL].to_numpy(), test_df[BASELINE_COL].to_numpy())
    print(f"[SCAFFOLD-DISJOINT HOLDOUT n={len(test_df)}, single-XGB] "
          f"MAE={holdout_xgb['MAE']:.3f} R2={holdout_xgb['R2']:.3f}  "
          f"(g-xTB baseline MAE={base_gxtb_holdout['MAE']:.3f})")

    # --- held-out test: MLP+XGB ensemble ---
    sc, mlp, xa, xb = _fit_ensemble(Xtr, ytr, args.seed)
    Xte_s = sc.transform(Xte)
    pred_ens = (mlp.predict(Xte_s) + xa.predict(Xte) + xb.predict(Xte)) / 3.0
    pred_test_ens = test_df[BASELINE_COL].to_numpy() + pred_ens
    holdout_ens = dc.metrics_vs_dft(test_df[TARGET_COL].to_numpy(), pred_test_ens)
    print(f"[SCAFFOLD-DISJOINT HOLDOUT n={len(test_df)}, MLP+XGB ensemble] "
          f"MAE={holdout_ens['MAE']:.3f} R2={holdout_ens['R2']:.3f}")

    # --- save artifacts ---
    joblib.dump(m_xgb, args.outdir / "models" / "champion_scaffold_disjoint.joblib")
    ens_model = MLPXGBEnsemble(sc, mlp, xa, xb, medians, feats)
    joblib.dump(ens_model, args.outdir / "models" / "ensemble_scaffold_disjoint.joblib")
    (args.outdir / "models" / "feature_list.json").write_text(json.dumps(feats, indent=2))

    metadata = {
        "n_total": len(df), "n_clean_train": len(train_df), "n_clean_test": len(test_df),
        "n_excluded_validation": int((df["new_scaffold_split"] == "validation").sum()),
        "n_excluded_mixed": int((df["new_scaffold_split"] == "mixed").sum()),
        "n_features": len(feats),
        "clean_train_cv_xgb": cv_xgb,
        "scaffold_disjoint_holdout_xgb": holdout_xgb,
        "scaffold_disjoint_holdout_ensemble": holdout_ens,
        "scaffold_disjoint_holdout_gxtb_baseline": base_gxtb_holdout,
    }
    (args.outdir / "models" / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nwrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
