"""Unified inference featurization — IDENTICAL geometry treatment to training.

Training (pipeline/compute/featurize.py) computes the descriptors and the xTB ΔG
on ONE shared dmso-optimised best conformer. To avoid a train/inference geometry
mismatch, inference must do the same. This calls the very same vendored
`featurize_one` with the DFT step turned off (`sp_method=None`) and a single
conformer (`ensemble_k=1`), yielding the descriptors + `dG_xtb_kcal` on one shared
geometry — exactly the training feature space, minus the (label-only) ORCA step.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from . import _featurize_backend as _fb
from . import _thermo_backend as _th
from . import config


def featurize_inference(smiles: str, *, xtb_bin: str | None = None,
                        multiwfn_bin: str | None = None, n_confs: int = 10):
    """Return (feature_row, dG_xtb_kcal, benzoin_smiles) on one shared geometry.

    `feature_row` is a dict with all descriptor columns + dG_xtb_kcal (ready for
    features.build_vector). Requires xTB; Multiwfn optional.
    """
    xtb = config.find_xtb(xtb_bin)
    mwf = config.find_multiwfn(multiwfn_bin)
    if config.xtb_share() and "XTBPATH" not in os.environ:
        os.environ["XTBPATH"] = config.xtb_share()

    tmp = Path(tempfile.mkdtemp(prefix="benzoin_feat_"))
    (tmp / "xyz").mkdir(parents=True, exist_ok=True)
    try:
        row = _fb.featurize_one(
            {"index": "query", "SMILES": smiles}, 0,
            work_dir=tmp, xyz_dir=tmp / "xyz", xtb_bin=xtb, mwf_bin=mwf,
            do_multiwfn=bool(mwf), solvent="dmso", n_confs=n_confs,
            ensemble_k=1, ensemble_window=3.0,        # single best conformer
            sp_method=None, sp_basis="", orca_bin="",  # no DFT at inference
            orca_nprocs=1, orca_maxcore=1000, T=298.15, P_atm=1.0,
            max_atoms=0, xtb_cores=2, parallel_jobs=4,
        )
        return row, row.get("dG_xtb_kcal"), _th._make_benzoin_smiles(smiles)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
