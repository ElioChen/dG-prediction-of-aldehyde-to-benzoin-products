#!/usr/bin/env python
"""Build the pairs CSV for a small (~750-pair) exploratory batch of brand-new
cross pairs (2026-09-20, user: queue idle, grow cross data -- agreed scope:
uniform random, small scale). Uses FlyingDataset.sample() (2026-09-20,
step-6 pilot infra) to draw addresses never seen by the 35,136-pair labeled
set, then writes them in rec1_b973c_tierB_worker.py's expected schema so the
same proven self-consistent (conf_funnel_v3 + GFN2 --ohess + r2SCAN-3c/
B97-3c/g-xTB SP per species) recipe used for Tier B can label them --
genuinely new compute (product geometry has never been computed for these
pairs), unlike the homo success-rate fixes which mostly reused archives.

Usage:
    /home/schen3/venv/nequip/bin/python cross_benzoin/build_cross_explore_pairs.py --n 750
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chemical_space import FlyingDataset  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
KNOWN_PAIRS = REPO / "data/chemical_space/labeled_pairs.parquet"
OUT = REPO / "data/cross_benzoin/cross_explore_20260920/cross_explore_pairs.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    known = pd.read_parquet(KNOWN_PAIRS)
    fd = FlyingDataset(known_pairs=known)
    rows = fd.sample(args.n, seed=args.seed)
    print(f"sampled {len(rows)} never-labeled cross pairs (0 already computed_full: "
          f"{sum(r['computed_full'] for r in rows)})")

    out = pd.DataFrame({
        "pid": [f"explore_{r['donor_idx']}_{r['acceptor_idx']}" for r in rows],
        "donor_smiles": [r["donor_smiles"] for r in rows],
        "acceptor_smiles": [r["acceptor_smiles"] for r in rows],
        "prod_smiles": [r["product_smiles"] for r in rows],
        "grp": "cross_explore_20260920",
        "new_scaffold_split": None,
        "label": np.nan,
        "gxtb_stored": np.nan,
        "donor_idx": [r["donor_idx"] for r in rows],
        "acceptor_idx": [r["acceptor_idx"] for r in rows],
        "reaction_type": [r["reaction_type"] for r in rows],
    })
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}")
    print(out["reaction_type"].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
