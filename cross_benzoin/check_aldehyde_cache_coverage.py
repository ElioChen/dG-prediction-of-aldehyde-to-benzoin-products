#!/usr/bin/env python
"""
Pre-flight check: run BEFORE submitting any cb_featurize.py array with
--require-cache-complete. Reports which donor/acceptor molecules in a candidate pool
are missing from the aldehyde cache, so gaps can be resolved (or accepted) up front
instead of discovering them via a batch of failed array tasks after the fact.

Root cause this addresses (found 2026-07-20, round8): the aldehyde library
(220,859 molecules) and the pre-computed cache CSV it's checked against aren't
perfectly 1:1 (~334-molecule gap, e.g. aldehydes_all.csv has 220,525 entries) --
any freshly-drawn candidate pool has a small but nonzero chance of drawing one of
the uncached molecules. --require-cache-complete then hard-fails the WHOLE chunk
containing it (by design, to prevent a "should be cache-only" run silently
ballooning into fresh xTB compute) -- round8 lost 8/81 chunks this way and needed
a manual retry pass. Running this script first turns that into a known, bounded,
pre-launch number instead of a post-hoc failure investigation.

Usage:
  python cross_benzoin/check_aldehyde_cache_coverage.py \
      --pairs data/cross_benzoin/cross_round9/cross_round9_pairs.csv \
      --cache data/cross_benzoin/homo_v6/aldehydes_all.csv

If gaps are found, options (pick based on how many):
  - few (<20): resubmit those pairs' chunks with REQUIRE_CACHE_COMPLETE=0 after the
    main run (same fix applied to round8 today), OR just launch the whole array
    with REQUIRE_CACHE_COMPLETE=0 from the start once you know the gap count is small
    and bounded (accepted, tiny extra compute) instead of the default fail-loud gate.
  - many: investigate why (e.g. candidate pool drawing from outside the cached
    library) before proceeding.
"""
from __future__ import annotations

import argparse

import pandas as pd
from rdkit import Chem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--smiles-col", default="smiles", help="column name in --cache holding SMILES")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)
    cache = pd.read_csv(args.cache)
    cache_smi = set(
        s for s in (Chem.CanonSmiles(s) if Chem.MolFromSmiles(s) else None
                    for s in cache[args.smiles_col].dropna())
        if s is not None
    )
    print(f"cache: {len(cache)} rows, {len(cache_smi)} unique canonical SMILES")

    missing = set()
    for role in ("donor", "acceptor"):
        col = f"{role}_smiles"
        if col not in pairs.columns:
            continue
        for smi in pairs[col].dropna().unique():
            m = Chem.MolFromSmiles(smi)
            if not m:
                continue
            c = Chem.CanonSmiles(smi)
            if c not in cache_smi:
                missing.add(c)

    print(f"\ncandidate pool: {len(pairs)} pairs")
    print(f"molecules missing from cache: {len(missing)}")
    if missing:
        print("\nmissing SMILES (first 20):")
        for s in list(missing)[:20]:
            print(" ", s)
        print(f"\n{'few enough to accept inline (REQUIRE_CACHE_COMPLETE=0)' if len(missing) < 20 else 'investigate before proceeding'}")
    else:
        print("\nclean -- safe to launch with REQUIRE_CACHE_COMPLETE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
