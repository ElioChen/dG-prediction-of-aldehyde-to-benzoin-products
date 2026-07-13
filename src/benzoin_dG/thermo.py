"""Cheap xTB-level benzoin ΔG — the Δ-learning base term *and* a model feature.

Wraps the vendored thermo backend's `calc_thermo_one` with ORCA disabled
(``sp_method=None``), so only the fast xTB + RRHO path runs. The result
``dG_xtb_kcal`` is what the Δ-learning model corrects up to DFT level.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from . import config
from . import _thermo_backend as _b


def xtb_delta_g(
    smiles: str,
    *,
    xtb_bin: str | None = None,
    solvent: str = "dmso",
    n_confs: int = 10,
    temperature: float = 298.15,
    pressure_atm: float = 1.0,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """xTB ΔG for ``2 R-CHO -> benzoin``. Returns the backend row dict, which
    includes ``dG_xtb_kcal``, ``benzoin_smiles`` and ``error``."""
    xtb = config.find_xtb(xtb_bin)
    if xtb is None:
        return {"dG_xtb_kcal": None, "error": "xtb_not_found"}
    if config.xtb_share() and "XTBPATH" not in os.environ:
        os.environ["XTBPATH"] = config.xtb_share()

    tmp = tempfile.mkdtemp(prefix="benzoin_thermo_") if work_dir is None else str(work_dir)
    tmp = Path(tmp); tmp.mkdir(parents=True, exist_ok=True)

    rec = {"index": "query", "SMILES": smiles, "PubChem_CID": ""}
    try:
        return _b.calc_thermo_one(
            rec, 0, work_root=tmp, xtb_bin=xtb, shermo_bin=None,
            T=temperature, P_atm=pressure_atm, n_confs_max=n_confs,
            solvent=solvent, sp_method=None,      # ORCA OFF — xTB path only
        )
    finally:
        if work_dir is None:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
