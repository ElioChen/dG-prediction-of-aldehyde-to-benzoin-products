#!/usr/bin/env python3
"""
Round 9 candidate SELECTION — extends sample_round8_from_candidates_v3.py's
category-balanced pattern with two changes, both explicitly recommended in
the 2026-07-20 scale-up plan (cross_benzoin/docs/REPORT_session_recovery_and_
scaleup_plan_20260720_{ZH,EN}.md):

1. **Scale-up**: per-class target raised (default 1200 vs round8's 667) to reach
   the recommended 8,000-10,000-pair round size, given confirmed-ample CPU/GPU
   budget and a large (1.26M-pair) remaining eligible pool.
2. **Phosphorus-targeted stratum**: cross-five-diagnostics-20260717's finding
   that phosphorus-containing pairs are the single clearest error driver
   (MAE 4.14 vs 2.22 non-risk baseline) motivates deliberately oversampling
   phosphorus-tagged molecules on top of the 6 category-balanced draws,
   rather than waiting for uncertainty-based scoring to organically surface
   them (phosphorus is only ~0.63% of the library, so pure random/uncertainty
   sampling would under-represent it for a while).

Uses the PRODUCTION (80/10/10) scaffold-split eligible pool
(candidates_v3_pairs_with_scaffold_split.parquet — NOT the _721 archived
alternative, see memory cross-split-ratio-decision-20260720). Also excludes
round8's already-selected pairs (source_pair_id) so round9 doesn't
accidentally re-draw pairs round8 is already labeling.

This script does NOT run any xTB/g-xTB/DFT compute -- pool selection only.

Usage
  python cross_benzoin/sample_round9_from_candidates_v3.py \
      --per-class 1200 --n-phosphorus 800 --seed 43 \
      --exclude-pairs data/cross_benzoin/cross_round8/cross_round8_pairs.csv \
      --out data/cross_benzoin/cross_round9/cross_round9_pairs.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LOOKUP = REPO / "data/cross_benzoin/candidates_v3/candidates_v3_pairs_with_scaffold_split.parquet"
ALDEHYDES = REPO / "data/cross_benzoin/candidates_v3/aldehydes_with_scaffold_split.parquet"

OUTPUT_FIELDS = [
    "donor_id", "acceptor_id", "donor_smiles", "acceptor_smiles",
    "source_pair_id", "new_scaffold_split", "orientation", "class_pair", "stratum",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1200,
                     help="target unordered pairs per class_pair combo (x6 categories)")
    ap.add_argument("--n-phosphorus", type=int, default=800,
                     help="additional unordered pairs drawn from the phosphorus-tagged "
                          "stratum (donor OR acceptor is xtb_risk-tagged phosphorus), "
                          "on top of the 6 category strata")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--exclude-pairs", type=Path, nargs="*", default=[],
                     help="prior rounds' pairs CSVs (via source_pair_id) to exclude from "
                          "this draw, e.g. round8's, so rounds don't overlap")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    lookup = pd.read_parquet(LOOKUP)
    eligible = lookup[(lookup["new_scaffold_split"] == "train") & (~lookup["already_labeled"])].copy()
    print(f"lookup table: {len(lookup)} total candidate pairs")
    print(f"scaffold-train-eligible, not-yet-labeled pool: {len(eligible)} pairs")

    exclude_ids = set()
    for p in args.exclude_pairs:
        if p.exists():
            prior = pd.read_csv(p)
            if "source_pair_id" in prior.columns:
                exclude_ids |= set(prior["source_pair_id"].dropna().unique())
    if exclude_ids:
        before = len(eligible)
        eligible = eligible[~eligible["pair_id"].isin(exclude_ids)]
        print(f"excluded {before - len(eligible)} pairs already drawn by prior in-flight rounds")

    ald = pd.read_parquet(ALDEHYDES)[["InChIKey", "SMILES", "cho_class", "xtb_risk"]]
    ald["is_phosphorus"] = ald["xtb_risk"].fillna("").str.contains("phosphorus")

    eligible = eligible.merge(
        ald.rename(columns={"InChIKey": "donor_InChIKey", "SMILES": "donor_smiles",
                             "cho_class": "donor_class", "is_phosphorus": "donor_p"}),
        on="donor_InChIKey", how="left",
    ).merge(
        ald.rename(columns={"InChIKey": "acceptor_InChIKey", "SMILES": "acceptor_smiles",
                             "cho_class": "acceptor_class", "is_phosphorus": "acceptor_p"}),
        on="acceptor_InChIKey", how="left",
    )
    missing = eligible["donor_smiles"].isna().sum() + eligible["acceptor_smiles"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows missing a SMILES join -- dropping")
        eligible = eligible.dropna(subset=["donor_smiles", "acceptor_smiles"])

    eligible["class_pair"] = eligible.apply(
        lambda r: "__".join(sorted([r["donor_class"], r["acceptor_class"]])), axis=1
    )
    eligible["is_phosphorus_pair"] = eligible["donor_p"].fillna(False) | eligible["acceptor_p"].fillna(False)
    print(f"\nphosphorus-tagged pairs in eligible pool: {eligible['is_phosphorus_pair'].sum()} "
          f"({eligible['is_phosphorus_pair'].mean():.2%})")

    rng_seed = args.seed
    picked_frames = []

    # stratum 1: 6-way category-balanced (mirrors round8's method exactly)
    for cls, grp in eligible.groupby("class_pair"):
        n = min(args.per_class, len(grp))
        s = grp.sample(n=n, random_state=rng_seed)
        s = s.assign(stratum="category_balanced")
        picked_frames.append(s)
    cat_picked_ids = pd.concat(picked_frames)["pair_id"]

    # stratum 2: phosphorus-targeted, drawn from the REMAINING pool (no double-counting
    # against stratum 1) so it's a genuine addition, not a relabel of already-picked rows
    remaining = eligible[~eligible["pair_id"].isin(cat_picked_ids) & eligible["is_phosphorus_pair"]]
    n_p = min(args.n_phosphorus, len(remaining))
    if n_p < args.n_phosphorus:
        print(f"WARNING: only {len(remaining)} phosphorus pairs available after stratum-1 "
              f"exclusion, wanted {args.n_phosphorus}")
    p_picked = remaining.sample(n=n_p, random_state=rng_seed).assign(stratum="phosphorus_targeted")
    picked_frames.append(p_picked)

    picked = pd.concat(picked_frames, ignore_index=True)
    print(f"\npicked {len(picked)} unordered pairs total")
    print(picked["stratum"].value_counts().to_string())
    print(picked["class_pair"].value_counts().to_string())
    print(f"phosphorus-tagged fraction of final draw: {picked['is_phosphorus_pair'].mean():.2%} "
          f"(vs {eligible['is_phosphorus_pair'].mean():.2%} in the raw eligible pool)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in picked.iterrows():
        rows.append({
            "donor_id": r["donor_InChIKey"], "acceptor_id": r["acceptor_InChIKey"],
            "donor_smiles": r["donor_smiles"], "acceptor_smiles": r["acceptor_smiles"],
            "source_pair_id": r["pair_id"], "new_scaffold_split": r["new_scaffold_split"],
            "orientation": "A_donor_B_acceptor", "class_pair": r["class_pair"], "stratum": r["stratum"],
        })
        rows.append({
            "donor_id": r["acceptor_InChIKey"], "acceptor_id": r["donor_InChIKey"],
            "donor_smiles": r["acceptor_smiles"], "acceptor_smiles": r["donor_smiles"],
            "source_pair_id": r["pair_id"], "new_scaffold_split": r["new_scaffold_split"],
            "orientation": "B_donor_A_acceptor", "class_pair": r["class_pair"], "stratum": r["stratum"],
        })
    out_df = pd.DataFrame(rows, columns=OUTPUT_FIELDS)
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote {len(out_df)} directed rows ({len(picked)} unordered pairs) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
