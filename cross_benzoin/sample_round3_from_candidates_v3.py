#!/usr/bin/env python3
"""
Round 3 candidate sampling — DIRECTLY from candidates_v3's own 4M-row pair
list, not a fresh custom generator (see [[cross-benzoin-push-20260714]] /
[[cross-round2-active-learning]]: round 1 and round 2 both lost ~30-33% of
their frozen-holdout rows because custom-generated pairs don't respect
candidates_v3's molecule-disjoint train/validation/test split -- sampling
from candidates_v3's own 'train' split rows eliminates that loss by
construction, since every row it emits already carries a correct `split`).

candidates_v3's directed CSV is NOT globally shuffled -- it's laid out as 6
contiguous class_pair blocks per split (see manifest.json's pair_counts), each
~266,667 unordered pairs / 533,334 directed rows for the train split, with
both directed rows of a pair adjacent (confirmed: pair_id repeats for exactly
2 consecutive rows). This script streams the file ONCE, groups adjacent rows
by pair_id, and does systematic (stride) sampling within each of the 6
train-split class_pair blocks so the output pool stays balanced across
category-pairs -- same idea as round 1/2's pools, but every row now carries a
genuine, pre-computed molecule-disjoint split label, no custom generator
involved.

Excludes any pair already used in round 1 (cross_pilot_v1, 299 pairs) or
round 2 (cross_round2, 2098 pairs) so round 3 explores new chemistry.

Usage
  python cross_benzoin/sample_round3_from_candidates_v3.py \
      --per-class 700 --out data/cross_benzoin/cross_round3/cross_round3_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANDIDATES = REPO / "data/cross_benzoin/candidates_v3/cross_benzoin_dG_candidates_v3.csv.gz"
MANIFEST = REPO / "data/cross_benzoin/candidates_v3/manifest.json"
ROUND1_PRODUCTS = REPO / "data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_products.csv"
ROUND2_PRODUCTS = REPO / "data/cross_benzoin/cross_round2/cross_round2_products.csv"

OUTPUT_FIELDS = [
    "donor_id", "acceptor_id", "donor_smiles", "acceptor_smiles",
    "source_reaction_id", "source_pair_id", "split", "orientation", "class_pair",
]


def load_used_pairs() -> set[frozenset]:
    used: set[frozenset] = set()
    for path in (ROUND1_PRODUCTS, ROUND2_PRODUCTS):
        if not path.exists():
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
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    train_counts = {k.split("|", 1)[1]: v for k, v in manifest["pair_counts"].items()
                    if k.startswith("train|")}
    print("train split unordered-pair counts per class_pair:", train_counts)

    used = load_used_pairs()
    print(f"excluding {len(used)} already-used pairs (round1+round2)")

    stride = {c: max(1, n // args.per_class) for c, n in train_counts.items()}
    print("stride (pairs apart) per class_pair:", stride)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    picked = Counter()
    excluded = Counter()
    seen = Counter()

    with args.out.open("w", encoding="utf-8", newline="") as out_fh, \
         gzip.open(CANDIDATES, "rt", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        reader = csv.DictReader(fh)
        train_rows = (row for row in reader if row["split"] == "train")

        for pair_id, group in itertools.groupby(train_rows, key=lambda r: r["pair_id"]):
            pair_rows = list(group)
            if len(pair_rows) != 2:
                continue
            cls = pair_rows[0]["class_pair"]
            if picked[cls] >= args.per_class:
                continue
            idx = seen[cls]
            seen[cls] = idx + 1
            if idx % stride[cls] != 0:
                continue
            d, a = pair_rows[0]["donor_InChIKey"], pair_rows[0]["acceptor_InChIKey"]
            if frozenset((d, a)) in used:
                excluded[cls] += 1
                continue
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
            picked[cls] += 1

            if all(picked[c] >= args.per_class for c in train_counts):
                break

    print("picked per class_pair:", dict(picked))
    print("excluded (already-used) per class_pair:", dict(excluded))
    print(f"total pairs written: {sum(picked.values())} ({2 * sum(picked.values())} directed rows) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
