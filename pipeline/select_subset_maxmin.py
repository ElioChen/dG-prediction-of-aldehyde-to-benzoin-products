#!/usr/bin/env python
"""Principled ~1,500 training subset on the v6 library — category-stratified MaxMin.

Why not silhouette k-means: the v6 chemical space is continuous (silhouette ~0.05 at all k),
so k-means can't size or robustly pick a subset. The learning curve shows ~1,500 labels is
the plateau. MaxMin diversity gives better coverage than k-means medoids, and stratifying by
cho_class fixes MaxMin's known tendency to under-pick minority categories
([[maxmin-undersamples-aromatic]]). Allocation is proportional to the library composition so
the subset mirrors what we actually screen (ALL categories — no aromatic-only filter).

  python pipeline/select_subset_maxmin.py --n 1500
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

REPO = Path("/scratch-shared/schen3/benzoin-dg")
LIB = REPO / "data/library/aldehydes_clean_v6.csv"
OUT = REPO / "data/analysis/subset_v6"
CHO = "[CX3H1]=O"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT / "subset_v6_maxmin_1500.csv"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    patt = Chem.MolFromSmarts(CHO)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    df = pd.read_csv(LIB, low_memory=False)
    # keep valid, single-CHO
    mols, keep = [], []
    for i, s in enumerate(df["SMILES"]):
        m = Chem.MolFromSmiles(str(s))
        if m and len(m.GetSubstructMatches(patt)) == 1:
            mols.append(m); keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    print(f"valid single-CHO: {len(df)}")

    counts = df["cho_class"].value_counts()
    total = len(df)
    picks = []
    for cls, n_cls in counts.items():
        n_pick = max(1, round(args.n * n_cls / total))
        sub = df[df["cho_class"] == cls].reset_index()        # 'index' = position in df
        fps = [gen.GetFingerprint(Chem.MolFromSmiles(str(s))) for s in sub["SMILES"]]
        sel = MaxMinPicker().LazyBitVectorPick(fps, len(fps), min(n_pick, len(fps)), seed=args.seed)
        chosen = sub.iloc[list(sel)]
        picks.append(chosen)
        print(f"  {cls:16s} pool={n_cls:6d} -> picked {len(chosen)}")
    out = pd.concat(picks, ignore_index=True)
    cols = [c for c in ("name", "SMILES", "molecular_formula", "MW", "InChIKey",
                        "PubChem_CID", "cho_class", "xtb_risk") if c in out.columns]
    out[cols].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  n={len(out)}")
    print(out["cho_class"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
