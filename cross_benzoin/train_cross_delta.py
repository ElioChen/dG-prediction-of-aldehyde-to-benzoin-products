#!/usr/bin/env python3
"""
Phase 3 step 2 — first cross-benzoin Delta-model baseline.

Target:    y = dG_orca_kcal - dG_gxtb_kcal   (the DFT correction on top of g-xTB)
Predict:   dG_gxtb_kcal + model(X) ~= DFT-level cross-benzoin dG

g-xTB (not raw GFN2-xTB) is the pre-ML baseline here: job 24609263 showed raw
GFN2-xTB is unusable on cross products (MAE 15.7) while g-xTB alone already gets
MAE 3.45 zero-shot -- see REPORT_cross_pilot_dft_sp_validation_20260714_EN.md.

n=598 (299 unordered donor/acceptor pairs, both orientations kept). CV MUST be
grouped by unordered pair (pair_key) -- AB and BA share both parent molecules
and most descriptor blocks, so a plain per-row K-fold would leak molecule-level
signal across train/test (see CROSS_BENZOIN_ML_RECOMMENDATIONS.md's split table).
Repeated GroupKFold (several random_state reshuffles) damps split-noise at this
small n, matching train_delta.py's rationale for the homo model.

Also runs the required feature-block ablations from DESCRIPTOR_POLICY_CROSS.md:
2D-only, aldehydes-only (donor+acceptor), product-only, donor+acceptor,
all-raw-blocks, all+interaction (full).

Usage
  python cross_benzoin/train_cross_delta.py
  python cross_benzoin/train_cross_delta.py --folds 5 --repeats 20
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
from assemble_cross_training_table import ALDEHYDE_FEATS, PRODUCT_FEATS, RDKIT_FEATS  # noqa: E402

TABLE = REPO / "data/cross_benzoin/cross_pilot_v1/cross_train_table.parquet"
OUT = REPO / "data/cross_benzoin/cross_pilot_v1/train_v1"
BASELINE_COL = "dG_gxtb_kcal"
TARGET_COL = "dG_orca_kcal"
# Molecule-disjoint train/val/test split (SHA256(InChIKey)-based, 80/10/10,
# both aldehydes of a pair always in the same split) from the real 2M/4M
# candidate release codex built on GitHub -- pulled via git-lfs. Using it
# gives a genuine frozen held-out test set, satisfying NEXT_STEPS.md Phase 4's
# promotion-gate requirement, instead of only repeated pair-grouped CV.
SPLIT_MAP = REPO / "data/cross_benzoin/candidates_v3/inchikey_split_map.parquet"


def _feature_blocks(cols: list[str]) -> dict[str, list[str]]:
    """Split the assembled columns back into DESCRIPTOR_POLICY_CROSS.md's blocks,
    using the same column-name schemes assemble_cross_training_table.py wrote them
    with (donor_<ALDEHYDE_FEATS>/donor_<RDKIT_FEATS>, product_<RDKIT_FEATS>, bare
    PRODUCT_FEATS, interaction_*)."""
    twoD = ([f"donor_{c}" for c in RDKIT_FEATS if f"donor_{c}" in cols] +
            [f"acceptor_{c}" for c in RDKIT_FEATS if f"acceptor_{c}" in cols] +
            [f"product_{c}" for c in RDKIT_FEATS if f"product_{c}" in cols])
    interaction = [c for c in cols if c.startswith("interaction_")]
    donor_acc = ([f"donor_{c}" for c in ALDEHYDE_FEATS if f"donor_{c}" in cols] +
                [f"acceptor_{c}" for c in ALDEHYDE_FEATS if f"acceptor_{c}" in cols])
    product = [c for c in PRODUCT_FEATS if c in cols]
    product_2d = [f"product_{c}" for c in RDKIT_FEATS if f"product_{c}" in cols]
    # mordred: donor_/acceptor_ (free, full-library slim102) + product_ (cheap,
    # geometry-reuse compute) -- confirmed a real gain, MAE 2.965->2.921, see
    # assemble_cross_training_table_v3.py / [[cross-round3-and-ensemble72-packaging-20260714]].
    # Empty list (not an error) for tables assembled before mordred was folded in.
    mordred = [c for c in cols if "_mordred_" in c]
    raw_blocks = donor_acc + product + product_2d + [c for c in twoD if c not in product_2d]
    return {
        "2D_only": twoD,
        "aldehydes_only": donor_acc,
        "product_only": product + product_2d,
        "donor+acceptor": donor_acc + twoD,
        "all_raw_blocks": raw_blocks,
        "all+interaction": raw_blocks + interaction,
        "mordred_only": mordred,
        "all_raw_blocks+mordred": raw_blocks + mordred,
    }


def repeated_group_cv(df: pd.DataFrame, feats: list[str], groups: np.ndarray,
                      folds: int, repeats: int, seed: int):
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xdf.median(numeric_only=True)
    X = Xdf.fillna(medians)
    y = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()
    oof_sum = np.zeros(len(df))
    oof_cnt = np.zeros(len(df))
    for r in range(repeats):
        gkf = GroupKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        for tr, te in gkf.split(X, y, groups=groups):
            m = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3,
                                       "learning_rate": 0.05}, seed)
            m.fit(X.iloc[tr], y[tr])
            oof_sum[te] += m.predict(X.iloc[te])
            oof_cnt[te] += 1
    oof = oof_sum / np.maximum(oof_cnt, 1)
    pred_dft = df[BASELINE_COL].to_numpy() + oof
    delta = dc.metrics_vs_dft(df[TARGET_COL].to_numpy(), pred_dft)
    base_gxtb = dc.metrics_vs_dft(df[TARGET_COL].to_numpy(), df[BASELINE_COL].to_numpy())
    base_xtb = dc.metrics_vs_dft(df[TARGET_COL].to_numpy(), df["dG_xtb_kcal"].to_numpy())
    return delta, base_gxtb, base_xtb, pred_dft, medians


def pair_split_labels(df: pd.DataFrame, *, verbose: bool = True) -> pd.Series | None:
    """Per-row candidates_v3 split label ('train'/'validation'/'test'), or NaN
    if either aldehyde is missing from the split map or the two aldehydes
    disagree (shouldn't happen -- molecule-disjoint by construction -- but a
    custom pair generator that doesn't consult the split map can violate it).
    Shared by frozen_holdout_eval() and main()'s leakage-safe final fit, so
    there is exactly one definition of "which rows touch the held-out test/
    validation split" in this file."""
    if not SPLIT_MAP.exists():
        return None
    smap = pd.read_parquet(SPLIT_MAP).rename(columns={"InChIKey": "id_"})
    lut = dict(zip(smap["id_"], smap["split"]))
    donor_split = df["donor_id"].map(lut)
    acceptor_split = df["acceptor_id"].map(lut)
    mismatch = (donor_split != acceptor_split) & donor_split.notna() & acceptor_split.notna()
    if verbose and mismatch.any():
        print(f"  WARN: {int(mismatch.sum())} rows have donor/acceptor in different "
              f"candidates_v3 splits (shouldn't happen, molecule-disjoint by construction)")
    pair_split = donor_split.where(~mismatch)
    if verbose:
        n_missing = pair_split.isna().sum()
        if n_missing:
            print(f"  {n_missing} rows have an aldehyde not found in candidates_v3's split map -- dropped")
        print("  pair_split counts:", pair_split.value_counts().to_dict())
    return pair_split


def frozen_holdout_eval(df: pd.DataFrame, feats: list[str], seed: int):
    """Fit on candidates_v3's 'train'-split rows, evaluate once on its frozen
    'test'-split rows -- a genuine molecule-disjoint holdout, complementing
    (not replacing) the repeated pair-grouped CV above."""
    pair_split = pair_split_labels(df)
    if pair_split is None:
        return None

    train_mask = (pair_split == "train").to_numpy()
    test_mask = (pair_split == "test").to_numpy()
    if train_mask.sum() < 20 or test_mask.sum() < 5:
        print("  too few rows in train/test split for a frozen holdout fit -- skipped")
        return None

    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xdf[train_mask].median(numeric_only=True)
    X = Xdf.fillna(medians)
    y = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()
    m = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, seed)
    m.fit(X[train_mask], y[train_mask])
    pred_dft_test = df[BASELINE_COL].to_numpy()[test_mask] + m.predict(X[test_mask])
    delta = dc.metrics_vs_dft(df[TARGET_COL].to_numpy()[test_mask], pred_dft_test)
    base_gxtb = dc.metrics_vs_dft(df[TARGET_COL].to_numpy()[test_mask], df[BASELINE_COL].to_numpy()[test_mask])
    return {"n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "delta": delta, "gxtb_baseline": base_gxtb}


def make_parity(df, pred_dft, base_gxtb, delta, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    dG = df[TARGET_COL].to_numpy()
    lo, hi = float(min(dG.min(), pred_dft.min())), float(max(dG.max(), pred_dft.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.scatter(dG, df[BASELINE_COL], s=16, alpha=0.35, label=f"g-xTB (MAE {base_gxtb['MAE']:.2f})")
    ax.scatter(dG, pred_dft, s=16, alpha=0.7, label=f"Δ-learn (MAE {delta['MAE']:.2f})")
    ax.set_xlabel("dG_orca_kcal (r2SCAN-3c DFT, true)")
    ax.set_ylabel("predicted ΔG (kcal/mol)")
    ax.legend(); ax.set_title("Cross-benzoin ΔG — pair-grouped CV parity (n=598, 299 pairs)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def make_residual(df, pred_dft, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    resid = pred_dft - df[TARGET_COL].to_numpy()
    for cat, sub in df.groupby("reaction_type"):
        idx = sub.index
        ax.scatter(pred_dft[df.index.get_indexer(idx)], resid[df.index.get_indexer(idx)],
                  s=16, alpha=0.6, label=cat)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("predicted ΔG (kcal/mol)"); ax.set_ylabel("residual: pred − DFT (kcal/mol)")
    ax.set_title("Cross-benzoin Δ-model residuals by category pair")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def make_shap(model, X, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    expl = shap.Explainer(model, X)
    sv = expl(X)
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.table)
    print(f"loaded {len(df)} rows, {df['pair_key'].nunique()} unordered pairs")

    meta_cols = {"id", "donor_id", "acceptor_id", "pair_key", "reaction_type",
                "donor_smiles", "acceptor_smiles", "smiles",
                "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"}
    all_feats = [c for c in df.columns if c not in meta_cols]
    groups = df["pair_key"].to_numpy()

    blocks = _feature_blocks(all_feats)
    print(f"\n=== Feature-block ablations ({args.folds}x{args.repeats} grouped CV) ===")
    print(f"{'block':<18}{'n_feats':>8}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    ablation_rows = []
    for name, feats in blocks.items():
        if not feats:
            continue
        delta, base_gxtb, base_xtb, pred_dft, _ = repeated_group_cv(
            df, feats, groups, args.folds, args.repeats, args.seed)
        print(f"{name:<18}{len(feats):>8}{delta['MAE']:>8.3f}{delta['RMSE']:>8.3f}{delta['R2']:>8.3f}")
        ablation_rows.append({"block": name, "n_feats": len(feats), **delta})
    print(f"{'[baseline] g-xTB':<18}{'':>8}{base_gxtb['MAE']:>8.3f}{base_gxtb['RMSE']:>8.3f}{base_gxtb['R2']:>8.3f}")
    print(f"{'[baseline] xTB':<18}{'':>8}{base_xtb['MAE']:>8.3f}{base_xtb['RMSE']:>8.3f}{base_xtb['R2']:>8.3f}")
    pd.DataFrame(ablation_rows).to_csv(out / "data" / "ablations.csv", index=False)

    # Full model = "all_raw_blocks+mordred": interaction_* confirmed useless three
    # times over (n=598/2354/4120), mordred confirmed a real gain when present;
    # degrades gracefully to plain all_raw_blocks on tables built without mordred
    # (mordred list is then empty). Supersedes the old "all+interaction" default.
    feats = blocks["all_raw_blocks+mordred"]
    delta, base_gxtb, base_xtb, pred_dft, medians = repeated_group_cv(
        df, feats, groups, args.folds, args.repeats, args.seed)
    print(f"\n=== Full model ({len(feats)} features) — grouped CV vs {args.folds}x{args.repeats} ===")
    print(f"  g-xTB baseline   MAE={base_gxtb['MAE']:6.3f}  RMSE={base_gxtb['RMSE']:6.3f}  R2={base_gxtb['R2']:6.3f}")
    print(f"  xTB baseline     MAE={base_xtb['MAE']:6.3f}  RMSE={base_xtb['RMSE']:6.3f}  R2={base_xtb['R2']:6.3f}")
    print(f"  Δ-learning       MAE={delta['MAE']:6.3f}  RMSE={delta['RMSE']:6.3f}  R2={delta['R2']:6.3f}")
    print(f"  MAE improvement over g-xTB: {base_gxtb['MAE'] - delta['MAE']:+.3f} kcal/mol")

    print("\n=== Frozen molecule-disjoint holdout (candidates_v3 train/test split) ===")
    frozen = frozen_holdout_eval(df, feats, args.seed)
    if frozen:
        fd, fb = frozen["delta"], frozen["gxtb_baseline"]
        print(f"  n_train={frozen['n_train']}  n_test={frozen['n_test']}")
        print(f"  g-xTB baseline   MAE={fb['MAE']:6.3f}  RMSE={fb['RMSE']:6.3f}  R2={fb['R2']:6.3f}")
        print(f"  Δ-learning       MAE={fd['MAE']:6.3f}  RMSE={fd['RMSE']:6.3f}  R2={fd['R2']:6.3f}")

    # Final fit for SHIPPING / next active-learning round scoring -- must NOT
    # touch candidates_v3's held-out test/validation split, or the "frozen"
    # holdout stops being frozen for every artifact except frozen_holdout_eval's
    # own internal fit. Custom pair generators (rounds 1-2) don't consult the
    # split map, so a small fraction of their rows land in test/validation by
    # chance; exclude them here the same way frozen_holdout_eval already does.
    pair_split_final = pair_split_labels(df, verbose=False)
    if pair_split_final is not None:
        leak_mask = pair_split_final.isin(["test", "validation"]).to_numpy()
        if leak_mask.any():
            print(f"\nExcluding {leak_mask.sum()} rows in candidates_v3's test/validation split "
                  f"from the final shipped model (held out everywhere, not just frozen_holdout_eval)")
        clean_mask = ~leak_mask
    else:
        clean_mask = np.ones(len(df), dtype=bool)

    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    final_medians = Xdf[clean_mask].median(numeric_only=True)
    X_clean = Xdf[clean_mask].fillna(final_medians)
    y_clean = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()[clean_mask]
    final = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, args.seed)
    final.fit(X_clean, y_clean)
    # SHAP below still wants a full-schema X (fillna with the same medians used to fit).
    X = Xdf.fillna(final_medians)

    make_parity(df, pred_dft, base_gxtb, delta, out / "figs" / "parity.png")
    make_residual(df, pred_dft, out / "figs" / "residual_by_category.png")
    try:
        make_shap(final, X, out / "figs" / "importance.png")
    except Exception as e:
        print(f"(SHAP skipped: {e})")

    cv_csv = out / "data" / "cv_predictions.csv"
    dfo = df[["id", "donor_id", "acceptor_id", "pair_key", "reaction_type",
             BASELINE_COL, "dG_xtb_kcal", TARGET_COL]].copy()
    dfo["dG_pred"] = pred_dft
    dfo["abs_err_delta"] = (dfo["dG_pred"] - dfo[TARGET_COL]).abs()
    dfo["abs_err_gxtb"] = (dfo[BASELINE_COL] - dfo[TARGET_COL]).abs()
    dfo.to_csv(cv_csv, index=False)

    by_cat = dfo.groupby("reaction_type")[["abs_err_delta", "abs_err_gxtb"]].mean()
    by_cat.to_csv(out / "data" / "mae_by_category.csv")
    print("\nMAE by category pair (Δ-model vs g-xTB baseline):")
    print(by_cat.to_string())

    joblib.dump(final, out / "models" / "cross_delta_model.joblib")
    (out / "models" / "feature_list.json").write_text(json.dumps(feats, indent=2))
    (out / "models" / "metadata.json").write_text(json.dumps({
        "model": "xgb", "target": TARGET_COL, "baseline": BASELINE_COL,
        "n_samples": int(len(df)), "n_pairs": int(df["pair_key"].nunique()),
        "n_features": len(feats), "folds": args.folds, "repeats": args.repeats,
        "cv_delta_learning": delta, "cv_gxtb_baseline": base_gxtb, "cv_xtb_baseline": base_xtb,
        "frozen_holdout_candidates_v3": frozen,
        "n_final_fit": int(clean_mask.sum()),
        "n_excluded_test_validation_leakage": int((~clean_mask).sum()),
        "feature_medians": {k: float(v) for k, v in final_medians.items()},
        "ablations": ablation_rows,
    }, indent=2))

    print(f"\nSaved model + figs + metadata to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
