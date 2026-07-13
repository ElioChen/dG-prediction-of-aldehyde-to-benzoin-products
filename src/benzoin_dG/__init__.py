"""benzoin_dG — predict the Gibbs free energy of benzoin condensation
(2 R-CHO -> R-CH(OH)-C(=O)-R) from an aldehyde SMILES, via xTB + a Δ-learning
correction to DFT (PBE0-D4)."""
from __future__ import annotations

from .predict import Prediction, predict_dG, predict_dG_fast

__version__ = "0.1.0"
__all__ = ["predict_dG", "predict_dG_fast", "Prediction", "__version__"]
