#!/usr/bin/env python3
"""
Targeted AROMATIC expansion. Categorize the cleaned aldehyde pool by what the CHO
carbon is bonded to (the chemically decisive axis for the benzoin condensation),
report the counts, then MaxMin-pick a fraction of the carbo-aromatic class
(classic benzaldehyde-like substrate, lowest model error, currently only ~11% of
the labeled set) for the next featurize round — seeded on already-selected reps so
we ADD new diverse aromatics rather than re-pick.

Usage
  python pipeline/select_aromatic.py --frac 0.01
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parent.parent
CHO = Chem.MolFromSmarts("[CX3H1;+0](=[O;+0])[#6]")


def cho_class(mol) -> str:
    """Category from the CHO carbon's neighbour (aromatic-carbo / aromatic-hetero /
    vinyl-conjugated / aliphatic)."""
    hits = mol.GetSubstructMatches(CHO)
    if len(hits) != 1:
        return "none"
    nbr = hits[0][2]
    a = mol.GetAtomWithIdx(nbr)
    if a.GetIsAromatic():
        ri = mol.GetRingInfo()
        hetero = any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                     for r in ri.AtomRings() if nbr in r for i in r)
        return "aromatic_hetero" if hetero else "aromatic_carbo"
    if a.GetHybridization() == Chem.HybridizationType.SP2:
        return "vinyl_conj"
    return "aliphatic"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(REPO / "data/library/aldehydes_clean_v2.csv"))
    ap.add_argument("--existing", default=str(REPO / "data/library/subset_v5.csv"),
                    help="already-selected reps (seeds; avoid re-picking)")
    ap.add_argument("--target-class", default="aromatic_carbo",
                    choices=["aromatic_carbo", "aromatic_hetero", "aliphatic", "vinyl_conj"])
    ap.add_argument("--frac", type=float, default=0.01, help="fraction of target class to add")
    ap.add_argument("--n-add", type=int, default=None, help="absolute count (overrides --frac)")
    ap.add_argument("--method", default="hybrid", choices=["hybrid", "maxmin", "representative"],
                    help="hybrid = half density-random (typical) + half MaxMin (diverse edges)")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--out", default=str(REPO / "data/library/subset_aromatic_v1.csv"))
    ap.add_argument("--categorized-out",
                    default=str(REPO / "data/library/pool_categorized.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    pool = pd.read_csv(args.pool)
    smi_col = "SMILES" if "SMILES" in pool.columns else "smiles"
    print(f"Pool: {len(pool):,} from {Path(args.pool).name}")

    mols = [Chem.MolFromSmiles(s) for s in pool[smi_col]]
    pool["cho_class"] = [cho_class(m) if m is not None else "unparseable" for m in mols]
    print(f"Categorized [{time.time()-t0:.0f}s]:")
    for c, n in Counter(pool["cho_class"]).most_common():
        print(f"  {c:18s} {n:8,d}  ({100*n/len(pool):5.2f}%)")
    pool[["index", smi_col, "cho_class"]].to_csv(args.categorized_out, index=False)
    print(f"Wrote {args.categorized_out}")

    ca = pool[pool["cho_class"] == args.target_class].reset_index(drop=True)
    n_add = args.n_add if args.n_add else max(1, round(args.frac * len(ca)))
    print(f"\n{args.target_class} available: {len(ca):,}  -> select {n_add}")

    # fingerprints of the carbo-aromatic subset
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius, fpSize=args.n_bits)
    fps, keep = [], []
    for i, s in enumerate(ca[smi_col]):
        m = Chem.MolFromSmiles(s)
        if m is not None and not any(a.GetIsotope() for a in m.GetAtoms()):
            fps.append(gen.GetFingerprint(m)); keep.append(i)
    ca = ca.iloc[keep].reset_index(drop=True)
    pos_of_index = {idx: p for p, idx in enumerate(ca["index"])}

    # seed with already-selected carbo-aromatic reps so we ADD new ones
    seed_idx = set(pd.read_csv(args.existing)["index"]) if Path(args.existing).exists() else set()
    seed_pos = [pos_of_index[i] for i in seed_idx if i in pos_of_index]
    avail = [p for p in range(len(ca)) if p not in set(seed_pos)]
    print(f"Seeds (already-selected {args.target_class} in pool): {len(seed_pos)}; "
          f"available to pick: {len(avail)}")

    rng = np.random.default_rng(args.seed)
    repr_pos, maxmin_pos = [], []
    n_repr = n_add // 2 if args.method == "hybrid" else (n_add if args.method == "representative" else 0)
    n_maxmin = n_add - n_repr

    # (1) representative: a uniform random draw of carbo-aromatic ≈ density-proportional
    #     ("typical" benzaldehydes; KMeans-medoids are unstable on sparse fingerprints).
    if n_repr:
        repr_pos = [int(p) for p in rng.choice(avail, size=min(n_repr, len(avail)), replace=False)]
    # (2) MaxMin: seeded on existing reps + the representative draw, so it spreads into
    #     the diverse EDGES not already covered.
    if n_maxmin:
        picker = rdSimDivPickers.MaxMinPicker()
        first = [int(p) for p in seed_pos + repr_pos]
        picks = list(picker.LazyBitVectorPick(fps, len(fps), len(first) + n_maxmin,
                                              firstPicks=first, seed=args.seed))
        maxmin_pos = picks[len(first):]

    sel = ([(p, "representative") for p in repr_pos]
           + [(p, "maxmin") for p in maxmin_pos])
    new = ca.iloc[[p for p, _ in sel]].copy()
    new["sel_method"] = [m for _, m in sel]
    cols = [c for c in ["index", smi_col, "PubChem_CID", "sel_method"] if c in new.columns]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    new[cols].to_csv(args.out, index=False)
    print(f"Picked {len(new)} {args.target_class} ({len(repr_pos)} representative + "
          f"{len(maxmin_pos)} MaxMin) -> {args.out}")
    print(f"Done [{time.time()-t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
