#!/usr/bin/env python3
"""
Prune an assembled training table down to a saved run's exact feature list.

`train_cross_ensemble.py` picks its features by RULE (`_feature_blocks(...)
["all_raw_blocks+mordred"]`, which takes every column containing `_mordred_`),
not from a frozen list. The historical 7-round run scored 260 features because it
ran on a mordred-slimmed table; `assemble_cross_training_table_v3.py` emits the
un-slimmed mordred block, so the same rule picks 563 features there and the run
would not be comparable.

Pruning the table to the reference run's own `feature_list.json` first makes the
rule reproduce that list exactly (verified: selection is identical, not merely the
same length), which is what makes a rebuilt-vs-original CV comparison meaningful.

Usage:
    python cross_benzoin/prune_table_to_champion_features.py \
        --table data/cross_benzoin/cross_round7/cross_train_table_7rounds_recovered.parquet \
        --feature-list data/cross_benzoin/cross_round7/train_ensemble_7rounds_slim120_v1/models/feature_list.json \
        --out data/cross_benzoin/cross_round7/cross_train_table_7rounds_recovered_slim260.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# columns train_cross_delta/-ensemble need besides the features themselves
META = ["id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
        "donor_smiles", "acceptor_smiles", "smiles",
        "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal", "scaffold_split"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--feature-list", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    fl = json.loads(args.feature_list.read_text())
    fl = fl if isinstance(fl, list) else fl["features"]

    missing = [c for c in fl if c not in df.columns]
    if missing:
        print(f"ERROR: table is missing {len(missing)} of {len(fl)} listed features: "
              f"{missing[:10]}", file=sys.stderr)
        return 1

    meta = [c for c in META if c in df.columns]
    out = df[meta + fl].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{args.table.name}: {df.shape} -> {out.shape}")
    print(f"  kept {len(meta)} meta + {len(fl)} features -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
