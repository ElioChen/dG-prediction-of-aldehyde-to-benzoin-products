#!/usr/bin/env python3
"""Tier B (full self-consistent B97-3c relabel campaign) — build the pair list.

Emits ALL 35,528 labeled pairs of
  data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet
in the schema rec1_b973c_tierB_worker.py consumes
  pid, grp, new_scaffold_split, donor_smiles, acceptor_smiles, prod_smiles, label, gxtb_stored

Ordering: test -> validation -> (train+mixed, round-interleaved), so the frozen
scaffold-disjoint holdout is relabeled first and an early Delta-model A/B check is
possible after ~1 cluster-day.

  python select_rec1_b973c_tierB.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
TAB = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
OUTDIR = REPO / "data/cross_benzoin/rec1_b973c_tierB"
COLS = ["id", "donor_smiles", "acceptor_smiles", "smiles", "round",
        "new_scaffold_split", "dG_orca_kcal", "dG_gxtb_kcal"]
SPLIT_PRIORITY = {"test": 0, "validation": 1, "mixed": 2, "train": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TAB, columns=COLS)
    df = df[df["dG_orca_kcal"].notna()].copy()
    assert not df["id"].duplicated().any(), "duplicate pair ids"

    df["prio"] = df["new_scaffold_split"].map(SPLIT_PRIORITY).fillna(2).astype(int)
    # round-interleave within the bulk (prio 2) so any prefix is round-representative
    rng = df.sample(frac=1.0, random_state=a.seed).copy()
    rng["rk"] = rng.groupby(["prio", "round"]).cumcount()
    rng = rng.sort_values(["prio", "rk", "round"], kind="stable")

    out = rng.rename(columns={"id": "pid", "smiles": "prod_smiles",
                              "dG_orca_kcal": "label", "dG_gxtb_kcal": "gxtb_stored"})
    keep = ["pid", "grp_placeholder", "new_scaffold_split", "donor_smiles",
            "acceptor_smiles", "prod_smiles", "label", "gxtb_stored", "round"]
    out["grp_placeholder"] = out["new_scaffold_split"]
    out = out[keep].rename(columns={"grp_placeholder": "grp"})

    dst = OUTDIR / f"rec1_b973c_tierB_pairs_{len(out)}.csv"
    out.to_csv(dst, index=False)
    print(f"wrote {len(out)} pairs -> {dst}")
    print(out.groupby("new_scaffold_split").size().to_string())
    print("\nfirst 3 rows:")
    print(out.head(3).to_string())
    print(f"\ntest+validation are rows 0..{(out.new_scaffold_split.isin(['test','validation'])).sum()-1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
