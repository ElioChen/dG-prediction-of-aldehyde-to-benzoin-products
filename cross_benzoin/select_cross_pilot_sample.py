#!/usr/bin/env python3
"""Select a small, stratified, directed cross-benzoin pilot sample.

Draws donor/acceptor aldehyde pairs from the full clean_v6 library (not the
unlabeled 4M candidate release, which is behind Git LFS and not needed here:
clean_v6 is the exact same source pool candidates_v3 was built from). Only
picks molecules whose aldehyde-side GFN2/g-xTB free energy is ALREADY cached
in data/cross_benzoin/homo_v6/aldehydes_all.csv, so the pilot run only pays
for the genuinely new part -- the cross PRODUCT geometry/energy -- and can
run with --aldehyde-cache --require-cache-complete (zero aldehyde recompute).

Stratifies across the 6 unordered category-combinations of
{aromatic_carbo, aromatic_hetero, aliphatic} and expands every unordered pair
into 2 directed (donor, acceptor) rows, matching the schema
cb_featurize.py --pairs expects: donor_id,acceptor_id,donor_smiles,acceptor_smiles.

Usage:
  python select_cross_pilot_sample.py --n-per-combo 50 --seed 20260714 \
      --out ../data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CLEAN_V6 = HERE / "data" / "library" / "aldehydes_clean_v6.csv"
ALD_CACHE = HERE / "data" / "cross_benzoin" / "homo_v6" / "aldehydes_all.csv"
CATEGORIES = ["aromatic_carbo", "aromatic_hetero", "aliphatic"]


def load_cached_smiles(path: Path) -> set[str]:
    from rdkit import Chem
    cached = set()
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("xtb_optimized")).strip() != "True":
                continue
            if not (row.get("G_xtb") and row.get("G_gxtb")):
                continue
            mol = Chem.MolFromSmiles((row.get("smiles") or "").strip())
            if mol is not None:
                cached.add(Chem.MolToSmiles(mol, canonical=True))
    return cached


def load_library_by_category(path: Path, cached: set[str]) -> dict[str, list[dict]]:
    from rdkit import Chem
    by_cat: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for idx, row in enumerate(csv.DictReader(fh)):
            smi = (row.get("SMILES") or "").strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol, canonical=True)
            if canon not in cached:
                continue
            cat = row.get("cho_class") or "unknown"
            by_cat[cat].append({
                "id": row.get("InChIKey") or f"row{idx}",
                "smiles": canon,
                "xtb_risk": row.get("xtb_risk") or "",
            })
    return by_cat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-combo", type=int, default=50,
                     help="unordered pairs per category-combination (6 combos total)")
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean-v6", default=str(CLEAN_V6))
    ap.add_argument("--aldehyde-cache", default=str(ALD_CACHE))
    ap.add_argument("--exclude-pairs-csv", action="append", default=[],
                     help="prior pairs CSV(s) (donor_smiles,acceptor_smiles) to exclude, "
                          "so a follow-up batch doesn't recompute already-labeled pairs; "
                          "repeatable")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cached = load_cached_smiles(Path(args.aldehyde_cache))
    by_cat = load_library_by_category(Path(args.clean_v6), cached)
    for cat in CATEGORIES:
        print(f"  pool[{cat}] = {len(by_cat[cat])} (cache-hit molecules)")

    exclude: set[tuple[str, str]] = set()
    from rdkit import Chem as _Chem
    for excl_path in args.exclude_pairs_csv:
        with open(excl_path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    a = _Chem.MolToSmiles(_Chem.MolFromSmiles(row["donor_smiles"]), canonical=True)
                    b = _Chem.MolToSmiles(_Chem.MolFromSmiles(row["acceptor_smiles"]), canonical=True)
                except Exception:
                    continue
                exclude.add(tuple(sorted((a, b))))
    if exclude:
        print(f"  excluding {len(exclude)} already-sampled unordered pairs")

    combos = list(itertools.combinations_with_replacement(CATEGORIES, 2))
    unordered_pairs: list[tuple[dict, dict]] = []
    for cat_a, cat_b in combos:
        pool_a, pool_b = by_cat[cat_a], by_cat[cat_b]
        if not pool_a or not pool_b:
            continue
        seen = set(exclude)
        attempts = 0
        picked = 0
        while picked < args.n_per_combo and attempts < args.n_per_combo * 50:
            attempts += 1
            a = rng.choice(pool_a)
            b = rng.choice(pool_b)
            if a["smiles"] == b["smiles"]:
                continue  # no self-pairs -- homo is already covered separately
            key = tuple(sorted((a["smiles"], b["smiles"])))
            if key in seen:
                continue
            seen.add(key)
            unordered_pairs.append((a, b))
            picked += 1
        print(f"  combo {cat_a}/{cat_b}: {picked}/{args.n_per_combo} unordered pairs")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["donor_id", "acceptor_id",
                                            "donor_smiles", "acceptor_smiles"])
        w.writeheader()
        for a, b in unordered_pairs:
            w.writerow({"donor_id": a["id"], "acceptor_id": b["id"],
                        "donor_smiles": a["smiles"], "acceptor_smiles": b["smiles"]})
            w.writerow({"donor_id": b["id"], "acceptor_id": a["id"],
                        "donor_smiles": b["smiles"], "acceptor_smiles": a["smiles"]})
            n_rows += 2

    print(f"wrote {n_rows} directed rows ({len(unordered_pairs)} unordered pairs) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
