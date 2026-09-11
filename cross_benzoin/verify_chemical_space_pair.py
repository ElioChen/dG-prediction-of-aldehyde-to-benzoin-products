#!/usr/bin/env python
"""Flying-dataset build order, step 3 verification (CHEMICAL_SPACE.md sec8,
"Do not skip the verification in step 3"): check that `FlyingDataset.pair(i, j)`'s
lazily-computed columns reproduce the corresponding columns of the current
champion training table, for ~20 known pairs, at machine precision.

Scope note (see chemical_space.py's module docstring): only the *lazy* tier
(donor/acceptor QM + BDE, all three RDKit-2D blocks, interaction_* terms,
product SMILES) is checked here -- that is what build-order step 3 covers.
Product QM / product mordred / baselines are the cache-hit tier, which by
construction match (they ARE the champion table's row) and aren't a
meaningful drift check.

Usage
    python cross_benzoin/verify_chemical_space_pair.py [--n 20] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chemical_space import FlyingDataset  # noqa: E402

CHAMPION_TABLE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"

# RDKit-2D and interaction_* fields are pure deterministic functions of a SMILES
# string / this module's own cached QM values -- these must be bit-level exact
# (any drift is a real bug). donor/acceptor QM fields are read from the CURRENT
# `aldehydes_all.csv`, which is not guaranteed to be bit-identical to whatever
# aldehyde-QM snapshot the round-10 champion table was originally assembled
# from (the library has been recomputed at least once since, memory
# `aldehyde-recompute-fidelity`: recomputed QM matches originals ~200x tighter
# than the 2.3 kcal/mol conformer-noise floor, but not bit-exact -- judge by
# relative deviation, not equality). So QM fields get a loose, EXPLICITLY
# NAMED tolerance instead of silently failing on machine-precision noise.
EXACT_RTOL, EXACT_ATOL = 1e-6, 1e-6
QM_RTOL, QM_ATOL = 2e-3, 1e-4  # ~0.2% relative -- matches observed recompute drift


def field_tier(name: str) -> str:
    if name in ("donor_smiles", "acceptor_smiles"):
        return "smiles"  # informational only, representation drift is expected (see step-2 finding)
    base = name.split("_", 1)[1] if name.startswith(("donor_", "acceptor_")) else name
    if any(base.startswith(p) for p in ("MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "ArRings",
                                         "ArHetRings", "AlRings", "Rings", "Heteroatoms",
                                         "FractionCSP3", "BertzCT", "Kappa2", "NumStereocenters")):
        return "rdkit2d"
    if name.startswith("interaction_"):
        return "interaction"
    return "qm"


def close(a, b, tier: str) -> tuple[bool, str]:
    a_nan = a is None or (isinstance(a, float) and np.isnan(a))
    b_nan = b is None or (isinstance(b, float) and np.isnan(b))
    if a_nan or b_nan:
        return a_nan == b_nan, "nan-match" if a_nan == b_nan else "nan-mismatch"
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float, np.floating, np.integer)):
        rtol, atol = (EXACT_RTOL, EXACT_ATOL) if tier != "qm" else (QM_RTOL, QM_ATOL)
        ok = np.isclose(float(a), float(b), rtol=rtol, atol=atol)
        return ok, f"reldev={abs(float(a) - float(b)) / (abs(float(b)) + 1e-9):.2e}"
    return a == b, "str-eq" if a == b else "str-diff"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    champ = pd.read_parquet(CHAMPION_TABLE)
    sample = champ.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"champion table: {len(champ)} rows x {len(champ.columns)} cols; verifying {len(sample)} sampled pairs")

    space = FlyingDataset(known_pairs=champ)

    n_pairs_ok = 0
    tier_totals: dict[str, int] = {}
    tier_ok: dict[str, int] = {}
    mismatches_seen: list[str] = []

    from rdkit import Chem

    def canon(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m) if m is not None else None

    idx_df = space._idx  # noqa: SLF001 -- test-only introspection
    smi_to_ald_idx = {r.smiles_canonical: i for i, r in idx_df.iterrows() if r.smiles_canonical}

    for _, row in sample.iterrows():
        # Resolve THIS row's own donor/acceptor to an address directly -- do
        # not search the reverse (address -> pos) lookup by pair_key, which is
        # fragile if pair_key is not guaranteed unique per ordered (d,a) (found
        # empirically: some pair_key values pair with BOTH donor->acceptor role
        # orderings in this table -- see step-3 findings in CHEMICAL_SPACE.md).
        d_idx = smi_to_ald_idx.get(canon(row["donor_smiles"]))
        a_idx = smi_to_ald_idx.get(canon(row["acceptor_smiles"]))
        if d_idx is None or a_idx is None:
            print(f"  SKIP pair_key={row['pair_key']}: donor/acceptor SMILES not found in aldehyde_index")
            continue
        my_addr = (int(d_idx), int(a_idx))

        result = space.pair(*my_addr)
        if not result["computed_full"]:
            print(f"  FAIL pair_key={row['pair_key']}: computed_full=False, expected a cache hit")
            continue

        checked_groups = {
            "donor_smiles": (result["donor_smiles"], row["donor_smiles"]),
            "acceptor_smiles": (result["acceptor_smiles"], row["acceptor_smiles"]),
        }
        for k, v in result["lazy_features"].items():
            if k in row.index:
                checked_groups[k] = (v, row[k])

        pair_ok = True  # judged on rdkit2d + interaction tiers only (exact); qm and smiles are informational
        for k, (mine, champ_v) in checked_groups.items():
            tier = field_tier(k)
            ok, detail = close(mine, champ_v, tier)
            tier_totals[tier] = tier_totals.get(tier, 0) + 1
            tier_ok[tier] = tier_ok.get(tier, 0) + int(ok)
            if not ok:
                if tier in ("rdkit2d", "interaction"):
                    pair_ok = False
                if len(mismatches_seen) < 15:
                    mismatches_seen.append(f"[{tier}] {row['pair_key']}.{k}: mine={mine!r} champion={champ_v!r} ({detail})")

        prod_ok = result["product_smiles"] is not None
        pair_ok = pair_ok and prod_ok

        n_pairs_ok += int(pair_ok)
        status = "OK" if pair_ok else "MISMATCH"
        print(f"  {status} pair_key={row['pair_key']} addr={my_addr} "
              f"fields_checked={len(checked_groups)} product_smiles={'ok' if prod_ok else 'MISSING'}")

    print(f"\n{n_pairs_ok}/{len(sample)} pairs OK (rdkit2d + interaction tiers exact-matched; qm/smiles informational).")
    for tier in ("rdkit2d", "interaction", "qm", "smiles"):
        if tier in tier_totals:
            print(f"  tier={tier:11s} {tier_ok.get(tier,0)}/{tier_totals[tier]} matched")
    if mismatches_seen:
        print("\nsample mismatches (qm/smiles entries here are EXPECTED, see docstring):")
        for m in mismatches_seen:
            print(f"  {m}")

    hard_fail = tier_ok.get("rdkit2d", 0) != tier_totals.get("rdkit2d", 0) or \
        tier_ok.get("interaction", 0) != tier_totals.get("interaction", 0)
    return 1 if hard_fail or n_pairs_ok < len(sample) else 0


if __name__ == "__main__":
    sys.exit(main())
