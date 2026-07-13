#!/usr/bin/env python3
"""
Funnel v3 = funnel v2 + a TOPOLOGY GUARD.

The CREST A/B (findings_2026-06-14) showed the funnel's only *catastrophic* failure is
a BROKEN-connectivity lowest conformer: GFN-FF relaxes a conformer into an isomerised/
fragmented structure whose spuriously low xTB energy then drives the Boltzmann label
(idx 288552: funnel's lowest was broken and sat 38 kcal below the lowest intact one,
making ΔG 12 kcal too negative). CREST avoids this with an internal topology check —
but its GFN-FF metadynamics can UNDER-sample large complex benzoins where the funnel's
dense ETKDG finds lower minima. So instead of switching engines, give the funnel the
one thing it lacks: drop any conformer whose perceived heavy-atom topology differs from
the input graph, then rank what remains. Keeps v2's dense, deterministic sampling;
costs only a cheap RDKit bond perception per conformer.

Same interface as conf_funnel_v2.rank_conformers_funnel_v2.
"""
from __future__ import annotations

from pathlib import Path

import conf_funnel_v2 as v2
from conf_crest import _ref_topo, _xyz_topo  # shared topology fingerprint helpers


def rank_conformers_funnel_v3(
    smiles: str, work_dir: Path, xtb_bin: str,
    n_confs_max: int = 0, title: str = "", solvent: str = "",
    cores: int = 1, workers: int = 1, l10: int = 10,
) -> list[tuple[str, float]]:
    ranked = v2.rank_conformers_funnel_v2(
        smiles, work_dir, xtb_bin, n_confs_max=n_confs_max, title=title,
        solvent=solvent, cores=cores, workers=workers, l10=l10)
    if not ranked:
        return ranked
    ref = _ref_topo(smiles)
    clean = [(xyz, E) for xyz, E in ranked if _xyz_topo(xyz) == ref]
    # If perception flags ALL as broken (perception edge-case, not real breakage),
    # fall back to the unfiltered ranking rather than returning nothing.
    return clean if clean else ranked
