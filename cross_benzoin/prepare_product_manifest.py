#!/usr/bin/env python3
"""Stream directed aldehyde pairs into a validated cross-benzoin product manifest.

This is the cheap stage before conformer/QM work.  It accepts either the versioned
candidate release or a ``prepare_pair_chunks.py`` chunk and preserves direction:
donor carbonyl carbon becomes the product ketone carbon; acceptor carbonyl carbon
becomes the carbinol carbon.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

REACTION_SMARTS = (
    "[CX3H1:1](=[O:2]).[CX3H1:3](=[O:4])"
    ">>[C:1](=[O:2])[CH1:3]([OX2H1:4])"
)
PRODUCT_RXN = AllChem.ReactionFromSmarts(REACTION_SMARTS)
BENZOIN_CORE = Chem.MolFromSmarts("[CX3](=[OX1])[CH1]([OX2H1])")

OUTPUT_FIELDS = [
    "reaction_id", "pair_id", "split", "orientation", "class_pair",
    "donor_id", "donor_smiles", "acceptor_id", "acceptor_smiles",
    "product_smiles", "product_status", "product_error",
]


def open_read(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def open_write(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6)
    return path.open("w", encoding="utf-8", newline="")


def first(row: dict[str, str], *names: str) -> str:
    return next((str(row.get(name) or "").strip() for name in names if row.get(name)), "")


def build_product(donor_smiles: str, acceptor_smiles: str) -> tuple[str, str]:
    donor = Chem.MolFromSmiles(donor_smiles)
    acceptor = Chem.MolFromSmiles(acceptor_smiles)
    if donor is None:
        return "", "invalid_donor_smiles"
    if acceptor is None:
        return "", "invalid_acceptor_smiles"
    try:
        products = PRODUCT_RXN.RunReactants((donor, acceptor))
    except Exception as exc:
        return "", f"reaction_exception:{type(exc).__name__}"
    for product_tuple in products:
        product = product_tuple[0]
        try:
            Chem.SanitizeMol(product)
            if BENZOIN_CORE is None or not product.HasSubstructMatch(BENZOIN_CORE):
                continue
            return Chem.MolToSmiles(product, canonical=True, isomericSmiles=False), ""
        except Exception:
            continue
    return "", "no_sanitized_benzoin_product"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directed pair CSV or CSV.gz")
    parser.add_argument("--output", required=True, type=Path, help="Product manifest CSV or CSV.gz")
    parser.add_argument("--summary", type=Path, help="Optional JSON QC summary")
    parser.add_argument("--max-rows", type=int, default=0, help="Testing aid; 0 means all rows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    with open_read(args.input) as source, open_write(args.output) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(reader):
            if args.max_rows and index >= args.max_rows:
                break
            donor = first(row, "donor_SMILES", "donor_smiles")
            acceptor = first(row, "acceptor_SMILES", "acceptor_smiles")
            product, error = build_product(donor, acceptor)
            status = "valid" if product else "invalid"
            counts[status] += 1
            counts["rows"] += 1
            if error and len(examples) < 20:
                examples.append({"row": str(index + 2), "error": error})
            writer.writerow({
                "reaction_id": first(row, "reaction_id", "source_reaction_id") or str(index),
                "pair_id": first(row, "pair_id", "source_pair_id"),
                "split": first(row, "split"),
                "orientation": first(row, "orientation"),
                "class_pair": first(row, "class_pair"),
                "donor_id": first(row, "donor_InChIKey", "donor_id"),
                "donor_smiles": donor,
                "acceptor_id": first(row, "acceptor_InChIKey", "acceptor_id"),
                "acceptor_smiles": acceptor,
                "product_smiles": product,
                "product_status": status,
                "product_error": error,
            })
    summary = {
        "input": str(args.input), "output": str(args.output),
        "rows": counts["rows"], "valid": counts["valid"],
        "invalid": counts["invalid"],
        "valid_fraction": counts["valid"] / counts["rows"] if counts["rows"] else 0.0,
        "error_examples": examples,
        "reaction_smarts": REACTION_SMARTS,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not counts["invalid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
