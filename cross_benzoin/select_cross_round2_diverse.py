#!/usr/bin/env python3
"""
Round 2: diversity-sampled cross-benzoin candidates, clustered on the PRODUCT
(not just the donor/acceptor aldehydes).

Round 1 (`select_cross_pilot_sample.py`, job 24607515/24609263, 598 labeled
rows) picked donor/acceptor PAIRS uniform-at-random within each category
combo. That validated the pipeline (see
REPORT_cross_pilot_dft_sp_validation_20260714_{EN,ZH}.md) but leaves the
model information-starved: 598 rows across 6 category combos, with the two
category combos actually recorded as "aliph-aliph" (n=4) and "aliph-hetero"
(n=16) badly under-covered. Per-user direction: the fix is a bigger, more
scientifically sampled batch -- diversity ("clustering") over the actual
cross PRODUCTS, not just random pair draws.

Two-stage diversity design (kept tractable at ~220k aldehydes x millions of
possible directed pairs):
  1. MaxMin-pick a diverse sub-pool of aldehydes PER CATEGORY (radius-2 Morgan
     fingerprints) -- same proven method as select_subset_maxmin.py
     ([[maxmin-undersamples-aromatic]]: MaxMin beats k-means here, this
     project's chemical space has ~0.05 silhouette at every k).
  2. Within each of the 6 unordered category-pair combos, draw a bounded
     random candidate set from the diverse sub-pools, generate the ACTUAL
     product SMILES (FP.build_product -- the same, now formyl-bug-fixed,
     reaction cb_featurize.py uses for real compute), fingerprint the
     products, and MaxMin-pick the final target count by PRODUCT diversity.

Categories are classified with featurize_product.classify() (whole-molecule
aromaticity), the SAME function that determines the recorded reaction_type
at compute time -- round 1's stratification used clean_v6's `cho_class`
(CHO-local-environment classification), a DIFFERENT scheme, which is why its
"even" per-combo targets landed wildly uneven once actually recorded.

Usage
  python cross_benzoin/select_cross_round2_diverse.py \
      --out ../data/cross_benzoin/cross_round2/cross_round2_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "pipeline"))
sys.path.insert(0, str(HERE / "pipeline" / "compute"))
import featurize_product as FP  # noqa: E402

CLEAN_V6 = HERE / "data" / "library" / "aldehydes_clean_v6.csv"
ALD_CACHE = HERE / "data" / "cross_benzoin" / "homo_v6" / "aldehydes_all.csv"
CATEGORIES = ["aliphatic", "aromatic_carbo", "aromatic_hetero"]

FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Rebalanced targets (unordered pairs) -- deliberately oversample the combos
# round 1 starved (aliph-aliph had 2 unordered pairs, aliph-hetero had 8) so
# this round both grows the total and fixes the coverage gap flagged in the
# validation report. carbo/hetero combos still get a healthy increase too.
DEFAULT_TARGETS = {
    ("aliphatic", "aliphatic"): 300,
    ("aliphatic", "aromatic_carbo"): 300,
    ("aliphatic", "aromatic_hetero"): 300,
    ("aromatic_carbo", "aromatic_carbo"): 400,
    ("aromatic_carbo", "aromatic_hetero"): 400,
    ("aromatic_hetero", "aromatic_hetero"): 400,
}
CANDIDATE_MULTIPLIER = 20   # candidate pairs drawn per combo = target * this, before MaxMin-picking down
SUBPOOL_SIZE = 500          # diverse aldehydes kept per category before pairing


def load_cached_smiles(path: Path) -> set[str]:
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
            cat = FP.classify(canon)   # NOTE: whole-molecule scheme, matches compute-time labeling
            by_cat[cat].append({"id": row.get("InChIKey") or f"row{idx}", "smiles": canon})
    return by_cat


def load_exclude_pairs(paths: list[str]) -> set[tuple[str, str]]:
    exclude: set[tuple[str, str]] = set()
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    a = Chem.MolToSmiles(Chem.MolFromSmiles(row["donor_smiles"]), canonical=True)
                    b = Chem.MolToSmiles(Chem.MolFromSmiles(row["acceptor_smiles"]), canonical=True)
                except Exception:
                    continue
                exclude.add(tuple(sorted((a, b))))
    return exclude


def maxmin_subpool(pool: list[dict], size: int, seed: int) -> list[dict]:
    if len(pool) <= size:
        return pool
    fps = [FPGEN.GetFingerprint(Chem.MolFromSmiles(m["smiles"])) for m in pool]
    sel = MaxMinPicker().LazyBitVectorPick(fps, len(fps), size, seed=seed)
    return [pool[i] for i in sel]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--subpool-size", type=int, default=SUBPOOL_SIZE)
    ap.add_argument("--candidate-multiplier", type=int, default=CANDIDATE_MULTIPLIER)
    ap.add_argument("--exclude-pairs-csv", action="append", default=[])
    ap.add_argument("--clean-v6", default=str(CLEAN_V6))
    ap.add_argument("--aldehyde-cache", default=str(ALD_CACHE))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cached = load_cached_smiles(Path(args.aldehyde_cache))
    by_cat = load_library_by_category(Path(args.clean_v6), cached)
    print("cache-hit pool sizes by category (featurize_product.classify scheme):")
    for cat in CATEGORIES:
        print(f"  {cat:16s} {len(by_cat[cat])}")

    exclude = load_exclude_pairs(args.exclude_pairs_csv)
    print(f"excluding {len(exclude)} already-sampled unordered pairs" if exclude else "no exclusions")

    print(f"\nMaxMin diverse sub-pools (cap {args.subpool_size}/category):")
    subpool: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        subpool[cat] = maxmin_subpool(by_cat[cat], args.subpool_size, args.seed)
        print(f"  {cat:16s} {len(by_cat[cat])} -> {len(subpool[cat])} diverse")

    all_rows: list[dict] = []
    seen_products: set[str] = set()
    summary = []
    for (cat_a, cat_b), target in DEFAULT_TARGETS.items():
        pool_a, pool_b = subpool[cat_a], subpool[cat_b]
        same_cat = cat_a == cat_b
        n_candidates = target * args.candidate_multiplier
        tried: set[tuple[str, str]] = set()
        picked_pairs: list[tuple[dict, dict, str]] = []   # (a, b, product_smiles)
        attempts, max_attempts = 0, n_candidates * 3
        while len(tried) < n_candidates and attempts < max_attempts:
            attempts += 1
            a = rng.choice(pool_a)
            b = rng.choice(pool_b)
            if a["smiles"] == b["smiles"]:
                continue
            key = tuple(sorted((a["smiles"], b["smiles"])))
            if key in exclude or key in tried:
                continue
            tried.add(key)
            prod = FP.build_product(a["smiles"], b["smiles"])
            if not prod or prod in seen_products:
                continue
            picked_pairs.append((a, b, prod))

        if len(picked_pairs) <= target:
            final = picked_pairs
        else:
            fps = [FPGEN.GetFingerprint(Chem.MolFromSmiles(p)) for _, _, p in picked_pairs]
            sel = MaxMinPicker().LazyBitVectorPick(fps, len(fps), target, seed=args.seed)
            final = [picked_pairs[i] for i in sel]

        for a, b, prod in final:
            seen_products.add(prod)
            all_rows.append({"donor_id": a["id"], "acceptor_id": b["id"],
                             "donor_smiles": a["smiles"], "acceptor_smiles": b["smiles"]})
            all_rows.append({"donor_id": b["id"], "acceptor_id": a["id"],
                             "donor_smiles": b["smiles"], "acceptor_smiles": a["smiles"]})
        summary.append((f"{cat_a}/{cat_b}", target, len(tried), len(picked_pairs), len(final)))
        print(f"  combo {cat_a}/{cat_b}: target={target} candidates_tried={len(tried)} "
              f"valid_products={len(picked_pairs)} final_picked={len(final)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["donor_id", "acceptor_id", "donor_smiles", "acceptor_smiles"])
        w.writeheader()
        w.writerows(all_rows)

    n_pairs = len(all_rows) // 2
    print(f"\nwrote {len(all_rows)} directed rows ({n_pairs} unordered pairs) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
