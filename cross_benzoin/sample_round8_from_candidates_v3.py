#!/usr/bin/env python3
"""
Round 8 candidate SELECTION — draws from candidates_v3's remaining pool,
filtered through the scaffold-disjoint-rebuild lookup table
(`candidates_v3_pairs_with_scaffold_split.parquet`, produced 2026-07-17,
see memory `cross-scaffold-disjoint-rebuild-20260717` step 4) so that round8
only touches pairs that are:

  (a) NOT yet labeled (the lookup's own `already_labeled` column, cross-
      referenced against round1-7's 16,241 labeled pairs — verified here to
      match exactly: 15,063 of round1-7's pairs are inside candidates_v3
      proper, the other 1,178 came from outside sources like screen10k/pilot
      generators and are structurally absent from this table, correctly).
  (b) on the scaffold-TRAIN side of the new, genuinely scaffold-disjoint
      split (`new_scaffold_split == "train"`), never `test`/`validation`/
      `mixed` — this is the whole point of filtering before any round8
      sampling: earlier rounds (1-7) sampled from candidates_v3's OLD
      molecule-level split's `train` label, which the five-diagnostics
      session found leaks scaffolds into the old 29-row frozen holdout
      severely (93% leaked). Restricting round8 to the NEW scaffold-train
      side keeps the new, honest scaffold-disjoint held-out test/validation
      sets (n=450/483 in round1-7's own data, or the ~21k/21k reserved in
      candidates_v3 proper) uncontaminated for future rounds.

This mirrors `sample_round3_from_candidates_v3.py`'s per-class-pair stride
sampling (balanced across the 6 donor/acceptor category combinations), but
draws from the pre-filtered eligible pool instead of streaming the raw 4M-row
candidates CSV directly, and joins donor/acceptor SMILES + cho_class back in
from `aldehydes_with_scaffold_split.parquet` (the lookup table itself only
carries InChIKeys, not SMILES).

This script does NOT run any xTB/g-xTB/DFT compute — it only selects and
writes a pairs CSV in the same format `cb_featurize.py` / prior rounds'
pairs.csv files use (`donor_id,acceptor_id,donor_smiles,acceptor_smiles,...`).
Featurization (which fuses xTB conformer search + g-xTB SP, real compute) and
DFT-SP labeling are explicitly NOT run by this script or invoked anywhere in
this task — see the round8 section of STATUS_{EN,ZH}.md for the exact
follow-up commands, pending user approval.

Usage
  python cross_benzoin/sample_round8_from_candidates_v3.py \
      --per-class 667 --seed 42 \
      --out data/cross_benzoin/cross_round8/cross_round8_pairs.csv
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
    "source_pair_id", "new_scaffold_split", "orientation", "class_pair",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=667,
                     help="target unordered pairs per class_pair combo (x6 = total pairs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    lookup = pd.read_parquet(LOOKUP)
    eligible = lookup[(lookup["new_scaffold_split"] == "train") & (~lookup["already_labeled"])].copy()
    print(f"lookup table: {len(lookup)} total candidate pairs")
    print(f"scaffold-train-eligible, not-yet-labeled pool: {len(eligible)} pairs")

    ald = pd.read_parquet(ALDEHYDES)[["InChIKey", "SMILES", "cho_class"]]
    eligible = eligible.merge(
        ald.rename(columns={"InChIKey": "donor_InChIKey", "SMILES": "donor_smiles", "cho_class": "donor_class"}),
        on="donor_InChIKey", how="left",
    ).merge(
        ald.rename(columns={"InChIKey": "acceptor_InChIKey", "SMILES": "acceptor_smiles", "cho_class": "acceptor_class"}),
        on="acceptor_InChIKey", how="left",
    )
    missing = eligible["donor_smiles"].isna().sum() + eligible["acceptor_smiles"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows missing a SMILES join -- dropping")
        eligible = eligible.dropna(subset=["donor_smiles", "acceptor_smiles"])

    eligible["class_pair"] = eligible.apply(
        lambda r: "__".join(sorted([r["donor_class"], r["acceptor_class"]])), axis=1
    )
    print("eligible pool class_pair distribution:")
    print(eligible["class_pair"].value_counts().to_string())

    rng_seed = args.seed
    picked_frames = []
    for cls, grp in eligible.groupby("class_pair"):
        n = min(args.per_class, len(grp))
        picked_frames.append(grp.sample(n=n, random_state=rng_seed))
    picked = pd.concat(picked_frames, ignore_index=True)
    print(f"\npicked {len(picked)} unordered pairs across {picked['class_pair'].nunique()} class_pairs")
    print(picked["class_pair"].value_counts().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in picked.iterrows():
        rows.append({
            "donor_id": r["donor_InChIKey"], "acceptor_id": r["acceptor_InChIKey"],
            "donor_smiles": r["donor_smiles"], "acceptor_smiles": r["acceptor_smiles"],
            "source_pair_id": r["pair_id"], "new_scaffold_split": r["new_scaffold_split"],
            "orientation": "A_donor_B_acceptor", "class_pair": r["class_pair"],
        })
        rows.append({
            "donor_id": r["acceptor_InChIKey"], "acceptor_id": r["donor_InChIKey"],
            "donor_smiles": r["acceptor_smiles"], "acceptor_smiles": r["donor_smiles"],
            "source_pair_id": r["pair_id"], "new_scaffold_split": r["new_scaffold_split"],
            "orientation": "B_donor_A_acceptor", "class_pair": r["class_pair"],
        })
    out_df = pd.DataFrame(rows, columns=OUTPUT_FIELDS)
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote {len(out_df)} directed rows ({len(picked)} unordered pairs) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
