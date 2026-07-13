"""g-xTB baseline + funnel_v3 geometry alignment for inference.

The shipped Δ-model (baseline = "gxtb_cosmo_dmso") is trained on g-xTB COSMO-DMSO
reaction ΔG sitting on the funnel_v3 conformer geometry. To match it at inference we
must (1) compute descriptors on the funnel_v3 geometry (not the lighter ETKDG ranker),
and (2) use the g-xTB COSMO-DMSO baseline rather than the GFN2 one.

Rather than re-vendor the funnel_v3 stack, this bridges to the training modules under
`pipeline/compute` (single research repo) so inference is byte-identical to labelling:
  - `align_funnel_v3()` monkeypatches the package thermo backend's conformer ranker to
    `conf_funnel_v3.rank_conformers_funnel_v3` (the descriptor/dG_xtb geometry then
    matches training).
  - `gxtb_baseline_dG(smiles)` reuses the validated `gxtb_baseline._species`
    (funnel_v3 → GFN2 ohess → g-xTB COSMO-DMSO SP) to return the g-xTB reaction ΔG.
"""
from __future__ import annotations

import sys
from pathlib import Path

# pipeline/compute lives next to the package in this repo (…/benzoin-dg/pipeline/compute)
_REPO = Path(__file__).resolve().parents[2]
_COMPUTE = _REPO / "pipeline" / "compute"
if _COMPUTE.is_dir() and str(_COMPUTE) not in sys.path:
    sys.path.insert(0, str(_COMPUTE))

HARTREE_TO_KCAL = 627.509474
_aligned = False


def available() -> bool:
    return _COMPUTE.is_dir()


def align_funnel_v3() -> bool:
    """Point the package thermo backend's ranker at funnel_v3 (idempotent).
    Returns True on success. After this, featurize_inference's descriptors + dG_xtb
    sit on the funnel_v3 geometry, matching the training labels."""
    global _aligned
    if _aligned:
        return True
    import conf_funnel_v3  # noqa: E402  (needs _COMPUTE on sys.path)
    from . import _thermo_backend as _th
    _th._rank_conformers = conf_funnel_v3.rank_conformers_funnel_v3
    _aligned = True
    return True


def gxtb_baseline_dG(smiles: str, *, scratch: Path | None = None) -> float | None:
    """g-xTB COSMO-DMSO reaction ΔG (kcal/mol) on the funnel_v3 geometry, or None.
    Reuses the exact training-baseline code so inference matches the labels."""
    import tempfile
    import shutil
    import gxtb_baseline as gb  # noqa: E402
    import thermo_orca as _to   # noqa: E402

    bz = _to._make_benzoin_smiles(smiles)
    if not bz:
        return None
    wd = Path(scratch or tempfile.mkdtemp(prefix="gxtb_base_"))
    try:
        a = gb._species(smiles, wd / "ald", "ald")
        b = gb._species(bz, wd / "bz", "bz")
        if "G_gxtb" not in a or "G_gxtb" not in b:
            return None
        return round((b["G_gxtb"] - 2.0 * a["G_gxtb"]) * HARTREE_TO_KCAL, 4)
    finally:
        if scratch is None:
            shutil.rmtree(wd, ignore_errors=True)
