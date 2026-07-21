#!/usr/bin/env python3
"""
Generalized version of sample_round3_from_candidates_v3.py — samples
directly from candidates_v3's own train-split pairs, excluding any pairs
already used by ANY number of prior rounds/pools (pass their products.csv
paths via --used-products, as many as needed) instead of hardcoding round1+2.

IMPORTANT: round 3's script used deterministic systematic stride sampling
(idx % stride == 0), which is fine for a FIRST round but breaks down for a
second-generation round drawing from the SAME pool with the SAME per-class
target: the stride hit-positions are identical every run, so round 4 with
the same --per-class as round 3 collided with round 3's own picks almost
completely (verified: 12/4200 pairs survived on the first attempt, not the
intended ~4200). Fixed here with per-class RESERVOIR sampling (Algorithm R)
using a random seed, so each invocation draws a genuinely different random
subset of the class block -- collision with prior rounds' small used-set is
then just background probability (~700/266667 per class), not a structural
certainty, and the explicit --used-products exclusion still catches it.

Usage (round 4, excluding rounds 1-3):
  python cross_benzoin/sample_from_candidates_v3.py --per-class 700 --seed 44 \
      --used-products data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_products.csv \
                       data/cross_benzoin/cross_round2/cross_round2_products.csv \
                       data/cross_benzoin/cross_round3/cross_round3_products.csv \
      --out data/cross_benzoin/cross_round4/cross_round4_pairs.csv

Usage (10k screening pool, excluding rounds 1-4):
  python cross_benzoin/sample_from_candidates_v3.py --per-class 1700 --seed 45 \
      --used-products <round1..4 products.csv...> \
      --out data/cross_benzoin/screen10k/screen10k_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANDIDATES = REPO / "data/cross_benzoin/candidates_v3/cross_benzoin_dG_candidates_v3.csv.gz"
MANIFEST = REPO / "data/cross_benzoin/candidates_v3/manifest.json"

OUTPUT_FIELDS = [
    "donor_id", "acceptor_id", "donor_smiles", "acceptor_smiles",
    "source_reaction_id", "source_pair_id", "split", "orientation", "class_pair",
]


def load_used_pairs(paths: list[Path]) -> set[frozenset]:
    used: set[frozenset] = set()
    for path in paths:
        if not path.exists():
            print(f"  WARN: {path} not found, skipping its exclusion")
            continue
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                d, a = row.get("donor_id"), row.get("acceptor_id")
                if d and a:
                    used.add(frozenset((d, a)))
    return used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=700,
                    help="target unordered pairs per class_pair combo (x6 = total pairs)")
    ap.add_argument("--used-products", type=Path, nargs="*", default=[],
                    help="products.csv files whose donor_id/acceptor_id pairs must be excluded")
    ap.add_argument("--seed", type=int, default=42,
                    help="reservoir-sampling seed -- use a DIFFERENT seed per round so successive "
                         "rounds don't draw the same random subset")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    train_counts = {k.split("|", 1)[1]: v for k, v in manifest["pair_counts"].items()
                    if k.startswith("train|")}
    print("train split unordered-pair counts per class_pair:", train_counts)

    used = load_used_pairs(args.used_products)
    print(f"excluding {len(used)} already-used pairs (from {len(args.used_products)} products files)")

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    excluded = Counter()
    seen_eligible = Counter()          # count of non-excluded pairs seen so far, per class
    reservoir: dict[str, list] = {c: [] for c in train_counts}

    with gzip.open(CANDIDATES, "rt", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        train_rows = (row for row in reader if row["split"] == "train")

        for pair_id, group in itertools.groupby(train_rows, key=lambda r: r["pair_id"]):
            pair_rows = list(group)
            if len(pair_rows) != 2:
                continue
            cls = pair_rows[0]["class_pair"]
            d, a = pair_rows[0]["donor_InChIKey"], pair_rows[0]["acceptor_InChIKey"]
            if frozenset((d, a)) in used:
                excluded[cls] += 1
                continue
            n = seen_eligible[cls] + 1
            seen_eligible[cls] = n
            res = reservoir[cls]
            if len(res) < args.per_class:
                res.append(pair_rows)
            else:
                j = rng.randrange(n)
                if j < args.per_class:
                    res[j] = pair_rows

    picked = Counter()
    with args.out.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for cls, res in reservoir.items():
            picked[cls] = len(res)
            for pair_rows in res:
                for row in pair_rows:
                    writer.writerow({
                        "donor_id": row["donor_InChIKey"],
                        "acceptor_id": row["acceptor_InChIKey"],
                        "donor_smiles": row["donor_SMILES"],
                        "acceptor_smiles": row["acceptor_SMILES"],
                        "source_reaction_id": row["reaction_id"],
                        "source_pair_id": row["pair_id"],
                        "split": row["split"],
                        "orientation": row["orientation"],
                        "class_pair": cls,
                    })

    print("picked per class_pair (reservoir sample):", dict(picked))
    print("excluded (already-used) per class_pair:", dict(excluded))
    print(f"total pairs written: {sum(picked.values())} ({2 * sum(picked.values())} directed rows) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
