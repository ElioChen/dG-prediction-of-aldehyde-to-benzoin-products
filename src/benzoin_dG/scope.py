"""Benzoin substrate scope — the model is trained and valid ONLY for aromatic
aldehydes (CHO on a carbo- or hetero-aromatic ring). Aliphatic aldehydes (α-H →
enolization/aldol) are poor benzoin substrates (low yield) and off-target, so they
are excluded from prediction. This is a cheap SMILES-only pre-filter applied before
the expensive featurization. Mirrors pipeline/cho_category (kept self-contained so
the shipped package has no pipeline dependency).
"""
from __future__ import annotations

from rdkit import Chem

_CHO = Chem.MolFromSmarts("[CX3H1;+0](=[O;+0])[#6]")
AROMATIC_SCOPE = {"aromatic_carbo", "aromatic_hetero"}


def cho_class(smiles: str) -> str:
    """aromatic_carbo | aromatic_hetero | vinyl_conj | aliphatic | none."""
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


def in_scope(smiles: str) -> bool:
    return cho_class(smiles) in AROMATIC_SCOPE
