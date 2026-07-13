"""Single-molecule featurization — the SAME descriptors used to train the model.

Wraps the vendored batch backend's `process_one`, so an aldehyde featurized here
is identical to one featurized in the training pipeline (train/inference parity).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from . import config
from . import _descriptors_backend as _b


def compute_descriptors(
    smiles: str,
    *,
    xtb_bin: str | None = None,
    multiwfn_bin: str | None = None,
    n_confs: int = 5,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the ~63 descriptors for one aldehyde SMILES.

    Returns the raw descriptor dict (with an ``error`` key, "" on success).
    xTB is required for the electronic block; Multiwfn is optional.
    """
    xtb = config.find_xtb(xtb_bin)
    mwf = config.find_multiwfn(multiwfn_bin)
    if config.xtb_share() and "XTBPATH" not in os.environ:
        os.environ["XTBPATH"] = config.xtb_share()

    tmp = tempfile.mkdtemp(prefix="benzoin_desc_") if work_dir is None else str(work_dir)
    tmp = Path(tmp)
    xyz_dir = tmp / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    rec = {"index": "query", "SMILES": smiles, "PubChem_CID": ""}
    try:
        return _b.process_one(
            rec, 0, xyz_dir=xyz_dir, work_dir=tmp,
            xtb_bin=xtb, mwf_bin=mwf,
            do_xtb_opt=bool(xtb), do_multiwfn=bool(mwf),
            n_confs=n_confs,
        )
    finally:
        if work_dir is None:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
