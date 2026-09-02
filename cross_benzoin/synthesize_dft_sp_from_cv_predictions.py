#!/usr/bin/env python3
"""
Reconstruct the per-round DFT-SP label files that the 2026-07 Snellius purge
destroyed, so the existing assembly scripts can run unmodified.

`data/raw/` was gitignored, so every `data/raw/dft_sp_cross/cross_round{N}/
cross_round{N}_dft_sp.csv` is gone from disk and from all three GitHub branches.
The labels themselves survive row-wise inside the training-run CV prediction
dumps, which ARE tracked: `cross_round7/unification_check_v1/
unified_cv_predictions.csv` carries `id`, `round` and `dG_orca_kcal` for 62,456
rows -- homo_unify_v1 (30,000) plus cross rounds 1-7 (32,456).

`load_round()` in assemble_cross_training_table_combined.py reads only
`id` and `dG_orca_kcal` (plus an optional `error`) out of the dft_sp file, so a
two-column reconstruction is a faithful stand-in for the label join.

Rounds 8 and 9 have NO surviving copy of their labels and cannot be rebuilt --
their DFT single points would have to be recomputed.

Usage:
    python cross_benzoin/synthesize_dft_sp_from_cv_predictions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CV = REPO / "data/cross_benzoin/cross_round7/unification_check_v1/unified_cv_predictions.csv"
OUT_ROOT = REPO / "data/raw/dft_sp_cross"
# round1 lives at a legacy path (see assemble_cross_training_table_combined.ROUND1_DFT)
LEGACY_OUT = {1: OUT_ROOT / "cross_pilot_v1_dft_sp.csv"}


def main() -> int:
    if not CV.exists():
        print(f"ERROR: label source not found: {CV}", file=sys.stderr)
        return 1
    cv = pd.read_csv(CV, usecols=["id", "round", "dG_orca_kcal"])
    print(f"label source: {len(cv)} rows, rounds {sorted(cv['round'].unique())}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    for n in range(1, 8):
        tag = f"round{n}"
        sub = cv[cv["round"] == tag]
        if sub.empty:
            print(f"  {tag}: no rows in label source -- skipped")
            continue
        sub = sub[["id", "dG_orca_kcal"]].dropna(subset=["dG_orca_kcal"]).drop_duplicates("id")
        out = LEGACY_OUT.get(n)
        if out is None:
            out = OUT_ROOT / f"cross_round{n}" / f"cross_round{n}_dft_sp.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
        total += len(sub)
        print(f"  {tag}: {len(sub)} labels -> {out.relative_to(REPO)}")
    print(f"wrote {total} reconstructed DFT-SP labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
