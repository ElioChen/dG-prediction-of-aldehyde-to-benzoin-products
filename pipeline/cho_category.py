#!/usr/bin/env python3
"""
Shared CHO-environment categorization — the chemically decisive axis for the
benzoin condensation (what the aldehyde carbon is bonded to). Used by the
targeted selectors, the per-category applicability domain, and (optionally) as
model features. Single source of truth so train/inference/selection agree.

  aromatic_carbo   CHO on a carbocyclic aromatic ring (benzaldehyde-like; classic)
  aromatic_hetero  CHO on a heteroaromatic ring (furfural, thiophene/pyridine-CHO)
  vinyl_conj       CHO on an sp2 C=C (α,β-unsaturated; acrolein-like)
  aliphatic        CHO on an sp3 carbon
"""
from __future__ import annotations

from rdkit import Chem

_CHO = Chem.MolFromSmarts("[CX3H1;+0](=[O;+0])[#6]")
CLASSES = ["aliphatic", "aromatic_carbo", "aromatic_hetero", "vinyl_conj"]
# one-hot feature columns (aliphatic is the reference / all-zero row)
FEATURE_COLS = ["cho_aromatic_carbo", "cho_aromatic_hetero", "cho_vinyl_conj"]


def cho_class(smiles) -> str:
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return "none"
    hits = m.GetSubstructMatches(_CHO)
    if len(hits) != 1:
        return "none"
    nbr = hits[0][2]
    a = m.GetAtomWithIdx(nbr)
    if a.GetIsAromatic():
        ri = m.GetRingInfo()
        hetero = any(m.GetAtomWithIdx(i).GetAtomicNum() != 6
                     for r in ri.AtomRings() if nbr in r for i in r)
        return "aromatic_hetero" if hetero else "aromatic_carbo"
    if a.GetHybridization() == Chem.HybridizationType.SP2:
        return "vinyl_conj"
    return "aliphatic"


def category_features(smiles_iter):
    """Return (one-hot DataFrame [FEATURE_COLS], list[str] of classes)."""
    import pandas as pd
    cls = [cho_class(s) for s in smiles_iter]
    df = pd.DataFrame({
        "cho_aromatic_carbo": [int(c == "aromatic_carbo") for c in cls],
        "cho_aromatic_hetero": [int(c == "aromatic_hetero") for c in cls],
        "cho_vinyl_conj": [int(c == "vinyl_conj") for c in cls],
    })
    return df, cls
