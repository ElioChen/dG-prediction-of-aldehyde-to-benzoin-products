"""Turn a raw descriptor dict + xTB ΔG into the exact model input vector.

The model ships with `feature_list.json` (column order) and `metadata.json`
(per-feature training medians). We reorder to that list and impute any missing
value with the training median — identical to what `pipeline/delta_core.py` does
at train time, so a feature vector built here matches the trained space exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

XTB_DG = "dG_xtb_kcal"
_MODELS = Path(__file__).resolve().parent / "models"


def load_feature_spec(models_dir: str | Path | None = None) -> tuple[list[str], dict[str, float]]:
    d = Path(models_dir) if models_dir else _MODELS
    feats = json.loads((d / "feature_list.json").read_text())
    meta = json.loads((d / "metadata.json").read_text())
    medians = meta.get("feature_medians", {})
    return feats, medians


def build_vector(descriptors: dict, dG_xtb: float | None,
                 feats: list[str], medians: dict[str, float]) -> np.ndarray:
    """Assemble a (1, n_features) array in `feats` order, median-imputing NaNs."""
    src = dict(descriptors)
    src[XTB_DG] = dG_xtb
    row = []
    for f in feats:
        v = src.get(f, None)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        if not np.isfinite(v):
            v = float(medians.get(f, 0.0))
        row.append(v)
    return np.asarray(row, dtype=float).reshape(1, -1)
