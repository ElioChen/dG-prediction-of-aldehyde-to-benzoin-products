"""Runtime discovery of the external binaries the featurizer needs.

xTB and Multiwfn are *not* pip-installable, so we locate them at runtime from
(in order): an explicit argument, an environment variable, then PATH / a few
known install locations. Inference needs xTB (for the ΔG feature and the xTB
electronic descriptors); Multiwfn is optional — without it the ADCH/QTAIM
columns are imputed to the training medians.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Env vars take precedence; then PATH; then these fallbacks (the HPC install).
_XTB_FALLBACKS = ["/home/schen3/xtb/bin/xtb"]
_MWF_FALLBACKS = ["/home/schen3/mutiwfn/Multiwfn_noGUI"]


def _resolve(explicit: str | None, env: str, names: list[str],
             fallbacks: list[str]) -> str | None:
    for cand in [explicit, os.environ.get(env)]:
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return str(Path(cand).resolve())
    for name in names:
        hit = shutil.which(name)
        if hit:
            return hit
    for fb in fallbacks:
        if Path(fb).is_file() and os.access(fb, os.X_OK):
            return fb
    return None


def find_xtb(explicit: str | None = None) -> str | None:
    return _resolve(explicit, "XTB_BIN", ["xtb"], _XTB_FALLBACKS)


def find_multiwfn(explicit: str | None = None) -> str | None:
    return _resolve(explicit, "MULTIWFN_BIN", ["Multiwfn_noGUI", "Multiwfn"],
                    _MWF_FALLBACKS)


def xtb_share() -> str | None:
    """XTBPATH (parameter dir) — xTB needs it set for some calculations."""
    p = os.environ.get("XTBPATH") or "/home/schen3/xtb/share/xtb"
    return p if Path(p).is_dir() else None
