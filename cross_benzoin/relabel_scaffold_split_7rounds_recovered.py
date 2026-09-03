#!/usr/bin/env python
"""Add pair-level `new_scaffold_split` to the recovered rounds-1-7 slim260 table so
the blend GNN (train_cross_gnn_arch_sweep.py) can retrain -- all cross GNN `.pt`
weights were lost in the 2026-07 purge (RECOVERY_REPORT_20260902 section 2).

The SMILES-keyed `candidates_v3/aldehydes_with_scaffold_split.parquet` the round8/9
relabelers used is gone, but the equivalent 80/10/10 Bemis-Murcko scaffold-disjoint
split survives numeric-id-keyed in
`data/cross_benzoin/homo_v6/aldehydes_scaffold_split_from_dG.csv`
(id = 0-based row index of aldehydes_clean_v6.csv; ids written float-formatted "2.0").

Rule (same as relabel_scaffold_split_{8,9}rounds.py): a pair is train/validation/test
only if donor AND acceptor land in the SAME molecule-level split; else "mixed"
(excluded from both train and test).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IN = REPO / "data/cross_benzoin/cross_round7/cross_train_table_7rounds_recovered_slim260.parquet"
OUT = REPO / "data/cross_benzoin/cross_round7/cross_train_table_7rounds_recovered_scaffold_split_labeled_slim260.parquet"
SPLIT_CSV = REPO / "data/cross_benzoin/homo_v6/aldehydes_scaffold_split_from_dG.csv"
CLEAN_V6 = REPO / "data/library/aldehydes_clean_v6.csv"


def main() -> int:
    sp = pd.read_csv(SPLIT_CSV)
    sp["id"] = sp["id"].astype(float).astype(int)
    id2split = dict(zip(sp["id"], sp["scaffold_split"]))
    ik2id = {ik: i for i, ik in enumerate(
        pd.read_csv(CLEAN_V6, usecols=["InChIKey"])["InChIKey"].astype(str))}

    df = pd.read_parquet(IN)
    print("table:", df.shape)

    def _split_for(ik: object) -> str | None:
        i = ik2id.get(str(ik))
        return id2split.get(i) if i is not None else None

    ds = df["donor_id"].map(_split_for)
    as_ = df["acceptor_id"].map(_split_for)
    matched = ds.notna() & as_.notna()
    same = ds == as_
    df["new_scaffold_split"] = ds.where(matched & same, "mixed")
    print(f"matched {int(matched.sum())}/{len(df)} pairs to the library split")
    vc = df["new_scaffold_split"].value_counts()
    print(vc.to_string())
    print((vc / len(df) * 100).round(1).to_string())

    df.to_parquet(OUT)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
