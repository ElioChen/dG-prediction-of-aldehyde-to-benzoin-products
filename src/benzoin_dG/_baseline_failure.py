"""Substructure veto for the g-xTB baseline-failure tail (zero training cost).

The champion is a Δ-learning model: ``dG = dG_gxtb + ML(descriptors)``. Its
residual tracks the g-xTB baseline error almost linearly (corr 0.888 on the
150-molecule hard tail), and on a handful of substructures the g-xTB COSMO-DMSO
baseline itself is broken by 10-17 kcal/mol -- far more than a smooth QM-feature
regression can undo. Those molecules do NOT reliably get a wide quantile-PI (the
ensemble can be confidently wrong), so the uncertainty router in
``_ensemble72_predict_subprocess.py`` misses them. This module force-routes them
to DFT regardless of the predicted uncertainty width.

Evidence: ``pipeline/analysis/notes/gxtb_baseline_failure_hardtail_20260902.md``
-- g-xTB baseline mean|err| by group on the hard tail: bare P 17.2, phosphine
oxide 16.2, sulfonyl 15.7, amide 14.4, imine 10.0; whole-library baseline
mean|err| ~4.3. Same root cause as the cross-dG "phosphorus is the biggest
error driver" finding and the nitro-BDE g-xTB failures: g-xTB is unreliable for
hypervalent P/S and strongly polar poly-functional systems.
"""
from __future__ import annotations

from rdkit import Chem

# (label, SMARTS, min_count). SMARTS mirror pipeline/analysis/build_hard_set.py's
# HARD_SMARTS so this veto and the hard-set taxonomy stay in sync.
#   - phosphorus: any P. The diagnosis' worst rows are bare P (17.2) and
#     phosphine oxide (16.2); `[#15]` subsumes P=O, phosphonium and bare P.
#   - sulfonyl: 15.7 kcal even at a single instance (n=37) -- systematic, route it.
#   - poly_amide: the note prescribes poly-amide; a single amide (14.4, but very
#     common) is left to the uncertainty router rather than force-routed here.
_VETO = (
    ("phosphorus", "[#15]", 1),
    ("sulfonyl", "[#16X4](=[OX1])(=[OX1])", 1),
    ("poly_amide", "[CX3](=O)[NX3]", 2),
)
_VETO_PATS = tuple((lbl, Chem.MolFromSmarts(s), n) for lbl, s, n in _VETO)


def baseline_failure_groups(*smiles: str) -> list[str]:
    """Return the sorted list of g-xTB-baseline-failure substructure labels hit by
    any of the given SMILES (aldehyde and/or product). Empty if none."""
    hits: set[str] = set()
    for smi in smiles:
        if not isinstance(smi, str) or not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for lbl, pat, need in _VETO_PATS:
            if pat is not None and len(m.GetSubstructMatches(pat)) >= need:
                hits.add(lbl)
    return sorted(hits)


def baseline_failure_reason(*smiles: str) -> str:
    """``'gxtb_baseline_failure:<g1,g2,...>'`` if any input SMILES hits a known
    g-xTB-baseline-failure substructure, else ``''``."""
    g = baseline_failure_groups(*smiles)
    return ("gxtb_baseline_failure:" + ",".join(g)) if g else ""
