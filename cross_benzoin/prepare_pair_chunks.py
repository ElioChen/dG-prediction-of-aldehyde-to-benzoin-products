#!/usr/bin/env python3
"""Stream the directed v2 candidate CSV(.gz) into cb_featurize-compatible chunks."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

REQUIRED = {
    "reaction_id",
    "pair_id",
    "split",
    "orientation",
    "class_pair",
    "donor_InChIKey",
    "donor_SMILES",
    "acceptor_InChIKey",
    "acceptor_SMILES",
}

OUTPUT_FIELDS = [
    "donor_id",
    "acceptor_id",
    "donor_smiles",
    "acceptor_smiles",
    "source_reaction_id",
    "source_pair_id",
    "split",
    "orientation",
    "class_pair",
]


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directed candidate CSV or CSV.gz")
    parser.add_argument("--out", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=0, help="Testing aid; 0 means no limit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    chunk_rows = 0
    chunk_index = -1
    counts_by_split: Counter[str] = Counter()
    counts_by_class: Counter[str] = Counter()

    with ExitStack() as stack:
        source = stack.enter_context(open_text(args.input))
        reader = csv.DictReader(source)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"input is missing required columns: {sorted(missing)}")

        output_handle = None
        writer = None
        for row in reader:
            if args.split != "all" and row["split"] != args.split:
                continue
            if args.max_rows and total >= args.max_rows:
                break
            if writer is None or chunk_rows >= args.chunk_size:
                if output_handle is not None:
                    output_handle.close()
                chunk_index += 1
                chunk_rows = 0
                chunk_path = args.out / f"pairs_{chunk_index:06d}.csv"
                output_handle = chunk_path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
                writer.writeheader()

            writer.writerow(
                {
                    "donor_id": row["donor_InChIKey"],
                    "acceptor_id": row["acceptor_InChIKey"],
                    "donor_smiles": row["donor_SMILES"],
                    "acceptor_smiles": row["acceptor_SMILES"],
                    "source_reaction_id": row["reaction_id"],
                    "source_pair_id": row["pair_id"],
                    "split": row["split"],
                    "orientation": row["orientation"],
                    "class_pair": row["class_pair"],
                }
            )
            total += 1
            chunk_rows += 1
            counts_by_split[row["split"]] += 1
            counts_by_class[row["class_pair"]] += 1

        if output_handle is not None:
            output_handle.close()

    metadata = {
        "source": str(args.input),
        "selected_split": args.split,
        "chunk_size": args.chunk_size,
        "rows": total,
        "chunks": chunk_index + 1,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_class": dict(sorted(counts_by_class.items())),
        "columns": OUTPUT_FIELDS,
    }
    (args.out / "index.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
