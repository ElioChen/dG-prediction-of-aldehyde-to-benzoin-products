#!/usr/bin/env python3
"""
Zero-new-compute error-driver diagnostics for the cross-benzoin Delta-model,
mirroring homo's screen_v6 functional-group analysis (never done for cross
specifically). Three checks, all pure re-analysis of already-saved predictions:

1. Functional-group / xtb_risk error driver: merge CV residuals with each
   aldehyde's cho_class + xtb_risk (from the clean_v6 library), see which
   donor/acceptor category combos have the worst error.
2. Uncertainty calibration: for every round that used bootstrap-ensemble
   uncertainty scoring (rounds 2-7), compare the PRE-label predicted std
   (from cross_roundN_scored.csv) against the ACTUAL post-label error (once
   DFT-SP landed) -- does higher predicted uncertainty really mean higher
   true error?
3. Directional (A->B vs B->A) asymmetry: split CV residuals by orientation
   and compare aggregate MAE -- sanity check that the model isn't secretly
   worse in one direction.

Usage:
  python cross_benzoin/analysis/cross_error_diagnostics.py \
      --cv-predictions data/cross_benzoin/cross_round6/train_6rounds_mordred_slim120_v1/data/cv_predictions.csv \
      --rounds 2 3 4 5 6 7 \
      --outdir data/cross_benzoin/cross_round6/error_diagnostics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
ALD_LIB = REPO / "data/library/aldehydes_clean_v6.csv"


def load_ald_lookup() -> pd.DataFrame:
    ald = pd.read_csv(ALD_LIB, usecols=["InChIKey", "cho_class", "xtb_risk"])
    return ald.drop_duplicates("InChIKey").set_index("InChIKey")


def round_paths(n: int):
    """Return (scored_csv_path, restrict_ids_path_or_None) for round n.

    Most rounds have their own dedicated `cross_roundN_scored.csv` (already
    scoped to that round's own candidate batch, safe to use unrestricted).
    Round 5 is a known exception: it drew its DFT-SP batch straight from the
    shared `screen10k` reservoir (`screen10k_scored.csv`) without ever
    writing its own `cross_round5_scored.csv` -- and that same reservoir file
    was ALSO the source rounds 6 and 7 drew their (non-overlapping) batches
    from later. So round 5 must be restricted to its own
    `cross_round5_dft_products.csv` ids, or it would silently pick up
    round 6/7's ids too (double-counted under the wrong round tag, since
    those rounds already have their own dedicated scored files).
    """
    rtag = f"cross_round{n}"
    if n == 5:
        scored = REPO / "data/cross_benzoin/screen10k/screen10k_scored.csv"
        restrict = REPO / "data/cross_benzoin/cross_round5/cross_round5_dft_products.csv"
        return scored, restrict
    scored = REPO / "data/cross_benzoin" / rtag / f"{rtag}_scored.csv"
    return scored, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-predictions", type=Path, required=True)
    ap.add_argument("--rounds", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cv = pd.read_csv(args.cv_predictions)
    ald = load_ald_lookup()

    # ---------- 1. functional-group / xtb_risk error driver ----------
    cv2 = cv.copy()
    cv2["donor_cho_class"] = cv2["donor_id"].map(ald["cho_class"])
    cv2["donor_xtb_risk"] = cv2["donor_id"].map(ald["xtb_risk"])
    cv2["acceptor_cho_class"] = cv2["acceptor_id"].map(ald["cho_class"])
    cv2["acceptor_xtb_risk"] = cv2["acceptor_id"].map(ald["xtb_risk"])
    cv2["class_pair"] = cv2[["donor_cho_class", "acceptor_cho_class"]].apply(
        lambda r: "__".join(sorted([str(r.iloc[0]), str(r.iloc[1])])), axis=1)
    cv2["either_xtb_risk"] = (cv2["donor_xtb_risk"].fillna("") != "") | \
                              (cv2["acceptor_xtb_risk"].fillna("") != "")

    by_class = cv2.groupby("class_pair")["abs_err_delta"].agg(["mean", "count"]).sort_values(
        "mean", ascending=False)
    by_class.to_csv(args.outdir / "error_by_class_pair.csv")
    print("\n=== MAE by donor/acceptor class-pair ===")
    print(by_class.to_string())

    by_risk = cv2.groupby("either_xtb_risk")["abs_err_delta"].agg(["mean", "count"])
    by_risk.to_csv(args.outdir / "error_by_xtb_risk.csv")
    print("\n=== MAE by xtb_risk flag (either side) ===")
    print(by_risk.to_string())

    # which individual risk tags are worst (donor or acceptor side, pooled)
    risk_rows = pd.concat([
        cv2[["donor_xtb_risk", "abs_err_delta"]].rename(columns={"donor_xtb_risk": "xtb_risk"}),
        cv2[["acceptor_xtb_risk", "abs_err_delta"]].rename(columns={"acceptor_xtb_risk": "xtb_risk"}),
    ], ignore_index=True)
    risk_rows = risk_rows[risk_rows["xtb_risk"].fillna("") != ""]
    by_tag = risk_rows.groupby("xtb_risk")["abs_err_delta"].agg(["mean", "count"]).sort_values(
        "mean", ascending=False)
    by_tag.to_csv(args.outdir / "error_by_risk_tag.csv")
    print("\n=== MAE by individual xtb_risk tag (donor+acceptor pooled) ===")
    print(by_tag.to_string())

    # ---------- 2. uncertainty calibration ----------
    calib_rows = []
    for n in args.rounds:
        p, restrict_p = round_paths(n)
        if not p.exists():
            print(f"[round{n}] scored file missing, skip: {p}")
            continue
        scored = pd.read_csv(p, usecols=lambda c: c in {
            "id", "dG_pred_correction_std", "dG_pred"})
        if restrict_p is not None:
            if not restrict_p.exists():
                print(f"[round{n}] restrict-ids file missing, skip: {restrict_p}")
                continue
            keep_ids = pd.read_csv(restrict_p, usecols=["id"])["id"]
            n_before = len(scored)
            scored = scored[scored["id"].isin(keep_ids)]
            print(f"[round{n}] restricted shared-pool scored file {p.name} "
                  f"{n_before} -> {len(scored)} rows via {restrict_p.name}")
        merged = scored.merge(cv[["id", "dG_orca_kcal", "dG_pred"]], on="id",
                               suffixes=("_prelabel", "_cv"), how="inner")
        if merged.empty:
            print(f"[round{n}] no overlap with cv_predictions, skip")
            continue
        merged["actual_abs_err"] = (merged["dG_pred_prelabel"] - merged["dG_orca_kcal"]).abs()
        merged["round"] = n
        calib_rows.append(merged[["id", "round", "dG_pred_correction_std", "actual_abs_err"]])
    if calib_rows:
        calib = pd.concat(calib_rows, ignore_index=True)
        calib["std_decile"] = pd.qcut(calib["dG_pred_correction_std"], 10, duplicates="drop",
                                       labels=False)
        by_decile = calib.groupby("std_decile").agg(
            mean_pred_std=("dG_pred_correction_std", "mean"),
            mean_actual_err=("actual_abs_err", "mean"),
            n=("actual_abs_err", "size"))
        by_decile.to_csv(args.outdir / "uncertainty_calibration_by_decile.csv")
        corr = calib["dG_pred_correction_std"].corr(calib["actual_abs_err"], method="spearman")
        print(f"\n=== Uncertainty calibration (n={len(calib)}, rounds={args.rounds}) ===")
        print(f"Spearman corr(predicted_std, actual_abs_err) = {corr:.4f}")
        print(by_decile.to_string())
        (args.outdir / "calibration_spearman_corr.txt").write_text(
            f"spearman_corr={corr:.4f}\nn={len(calib)}\nrounds={args.rounds}\n")
    else:
        print("\n=== Uncertainty calibration: NO DATA (no scored/cv overlap found) ===")

    # ---------- 3. directional (A->B vs B->A) asymmetry ----------
    cv3 = cv.copy()
    cv3["orientation"] = np.where(cv3["donor_id"] < cv3["acceptor_id"], "donor_lt_acceptor",
                                   "donor_gt_acceptor")
    by_orient = cv3.groupby("orientation")["abs_err_delta"].agg(["mean", "count"])
    by_orient.to_csv(args.outdir / "error_by_orientation.csv")
    print("\n=== MAE by orientation (sanity: should be ~symmetric) ===")
    print(by_orient.to_string())

    # per-pair A->B vs B->A residual gap (paired, more sensitive than the aggregate split)
    piv = cv3.pivot_table(index="pair_key", columns="orientation", values="abs_err_delta")
    piv = piv.dropna()
    if len(piv) > 0:
        gap = (piv["donor_lt_acceptor"] - piv["donor_gt_acceptor"]).abs()
        print(f"\nPer-pair |A->B err - B->A err|: mean={gap.mean():.3f}, "
              f"median={gap.median():.3f}, n_pairs={len(piv)}")
        gap.to_frame("abs_orientation_gap").to_csv(args.outdir / "per_pair_orientation_gap.csv")

    print(f"\nwrote diagnostics to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
