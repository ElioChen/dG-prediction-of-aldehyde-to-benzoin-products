"""End-to-end: aldehyde SMILES -> benzoin-condensation ΔG (kcal/mol).

    dG_pred = dG_xtb + model(descriptors, dG_xtb)

where `model` is the Δ-learning correction trained to reproduce r2SCAN-3c DFT
(CPCM/DMSO). The trained model + its feature spec ship in `benzoin_dG/models/`.

The Δ target is purely electronic: both ΔG levels share the same xTB-optimized
geometry and the same xTB RRHO thermal correction, so dG_orca − dG_xtb == ΔΔE_el.
"""
from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib

from . import _thermo_backend as _th
from . import features as _feat
from . import scope as _scope
from .applicability import ADReference
from .featurize import featurize_inference

_MODELS = Path(__file__).resolve().parent / "models"


@dataclass
class Prediction:
    smiles: str
    benzoin_smiles: str | None
    dG_pred: float | None          # Δ-learning ΔG  (≈ r2SCAN-3c DFT)
    dG_xtb: float | None           # cheap xTB baseline ΔG
    dG_correction: float | None    # model's DFT correction
    favorable: bool | None = None  # dG_pred < 0  => benzoin product downhill
    confidence: str = "unknown"    # high | medium | low (AD + flexibility tier)
    ad_flag: str = "unknown"       # in_domain | borderline | extrapolation
    ad_distance: float | None = None   # descriptor-space dist to nearest training molecule
    cho_class: str = ""            # aromatic_carbo | aromatic_hetero | aliphatic | vinyl_conj
    benzoin_relevant: bool = True  # False => off-target (aliphatic/vinyl); not predicted
    method: str = "delta"          # "delta" (xTB + Δ-correction) | "surrogate_2d" (no xTB)
    error: str = ""
    model_version: str = ""
    scope_note: str = "aromatic aldehyde homo-benzoin"

    def as_dict(self) -> dict:
        return asdict(self)


def _confidence(ad_flag: str, rotbonds: float | None) -> str:
    """Per-molecule confidence tier — the 'flag, don't delete' lever.

    Combines applicability-domain distance with conformational flexibility, which
    is the dominant residual driver (measured stratified OOF MAE on the funnel
    labels: 0–3 rotbonds 1.82, 4–5 2.34, 6–7 2.48, 8+ 3.20 kcal/mol). Flexible
    molecules are not removed from training; they are reported as lower-confidence.
    """
    score = {"in_domain": 2, "borderline": 1, "extrapolation": 0}.get(ad_flag, 1)
    rb = rotbonds if rotbonds is not None else 4
    if rb >= 8:        # the 3.20 kcal/mol flexible tail
        score -= 1
    elif rb >= 6:      # 2.48 kcal/mol — cap below "high"
        score = min(score, 1)
    return {2: "high", 1: "medium"}.get(score, "low")


@functools.lru_cache(maxsize=1)
def _load_model(models_dir: str | None = None):
    d = Path(models_dir) if models_dir else _MODELS
    mp = d / "delta_model.joblib"
    if not mp.exists():
        raise FileNotFoundError(
            f"No trained model at {mp}. Train one with the pipeline "
            "(pipeline/sweep_delta.py) and copy the result into "
            "src/benzoin_dG/models/.")
    model = joblib.load(mp)
    feats, medians = _feat.load_feature_spec(d)
    ad = ADReference.load(d)          # optional; None if not shipped
    baseline = "gfn2"
    model_version = "delta_model"
    mj = d / "metadata.json"
    if mj.exists():
        import json
        meta = json.loads(mj.read_text())
        baseline = meta.get("baseline", "gfn2")
        model_version = (
            f"{meta.get('model', 'model')}:n{meta.get('n_samples', '?')}:"
            f"f{meta.get('n_features', len(feats))}:{baseline}"
        )
    return model, feats, medians, ad, baseline, model_version


@functools.lru_cache(maxsize=1)
def _load_surrogate(models_dir: str | None = None):
    """Load the fast pure-SMILES 2D surrogate (ships separately from the Δ-model).

    Returns (model, feats, medians). Built by pipeline/train_surrogate.py --save."""
    import json
    d = Path(models_dir) if models_dir else _MODELS
    mp = d / "surrogate_model.joblib"
    if not mp.exists():
        raise FileNotFoundError(
            f"No 2D surrogate at {mp}. Ship one with "
            "`python pipeline/train_surrogate.py --save`.")
    model = joblib.load(mp)
    feats = json.loads((d / "surrogate_features.json").read_text())
    meta = json.loads((d / "surrogate_metadata.json").read_text())
    medians = meta.get("feature_medians", {})
    return model, feats, medians


def predict_dG(smiles: str, *, xtb_bin: str | None = None,
               multiwfn_bin: str | None = None, n_confs: int = 5,
               models_dir: str | None = None) -> Prediction:
    """Predict benzoin-condensation ΔG for one aldehyde SMILES.

    The model is scoped to AROMATIC aldehydes (carbo/hetero). Aliphatic and
    α,β-unsaturated aldehydes are off-target for the benzoin condensation and are
    short-circuited (no featurize, no prediction) with benzoin_relevant=False.
    """
    cls = _scope.cho_class(smiles)
    if cls not in _scope.AROMATIC_SCOPE:
        return Prediction(smiles, None, None, None, None, cho_class=cls,
                          benzoin_relevant=False,
                          error=f"out_of_scope:{cls} (model is aromatic-aldehyde only)")
    model, feats, medians, ad, baseline, model_version = _load_model(models_dir)

    # The g-xTB-baseline model needs the funnel_v3 geometry for the descriptors AND a
    # g-xTB COSMO-DMSO baseline ΔG — both identical to training (see _gxtb_inference).
    gxtb = baseline == "gxtb_cosmo_dmso"
    if gxtb:
        from . import _gxtb_inference as _gi
        if not _gi.available():
            return Prediction(smiles, None, None, None, None, cho_class=cls,
                              error=("gxtb_baseline_unavailable: this model needs the "
                                     "local pipeline/compute bridge for g-xTB COSMO-DMSO. "
                                     "Run from the full research checkout or use --fast."),
                              model_version=model_version)
        _gi.align_funnel_v3()

    # Descriptors + GFN2 dG_xtb on ONE shared dmso-opt geometry (identical to training).
    # With funnel_v3 active, n_confs=0 lets the funnel use its own dense size-aware cap
    # (training used n_confs_max=0); a fixed small n_confs would under-sample vs training.
    row, dG_xtb, bz = featurize_inference(smiles, xtb_bin=xtb_bin,
                                          multiwfn_bin=multiwfn_bin,
                                          n_confs=0 if gxtb else n_confs)
    if row.get("error"):
        return Prediction(smiles, bz, None, dG_xtb, None, cho_class=cls,
                          error=f"featurize:{row['error']}", model_version=model_version)
    # For the g-xTB model the baseline IS the g-xTB ΔG (the descriptor geometry above is
    # only used for the feature vector; the baseline feature slot carries g-xTB).
    if gxtb:
        dG_xtb = _gi.gxtb_baseline_dG(smiles)
    if dG_xtb is None:
        return Prediction(smiles, bz, None, None, None, cho_class=cls,
                          error=("gxtb_dG_failed: verify g-xTB runtime and XTB_BIN"
                                 if gxtb else
                                 "xtb_dG_failed: verify XTB_BIN points to an executable xTB"),
                          model_version=model_version)
    # NB: a positive xTB ΔG is NOT an error — benzoin condensation is genuinely
    # unfavourable for many aldehydes. The AD flag handles out-of-domain inputs.

    X = _feat.build_vector(row, dG_xtb, feats, medians)
    import pandas as pd
    correction = float(model.predict(pd.DataFrame(X, columns=feats))[0])
    ad_flag, ad_dist = "unknown", None
    if ad is not None:
        res = ad.score(X[0])
        ad_flag, ad_dist = res.flag, res.distance
    dG_pred = dG_xtb + correction
    return Prediction(smiles, bz, dG_pred, dG_xtb, correction,
                      favorable=bool(dG_pred < 0),
                      confidence=_confidence(ad_flag, row.get("RotBonds")),
                      ad_flag=ad_flag, ad_distance=ad_dist,
                      cho_class=cls, benzoin_relevant=True, method="delta",
                      model_version=model_version)


def predict_dG_fast(smiles: str, *, models_dir: str | None = None) -> Prediction:
    """FAST tier: pure-SMILES 2D surrogate ΔG — no xTB, no conformer search.

    Predicts r2SCAN-3c ΔG directly from geometry-free RDKit-2D descriptors. ~1 kcal
    less accurate than the Δ-model (CV ~2.9 vs ~2.2) but instant, so it can pre-screen
    100k+ candidates before spending xTB on the promising ones. Same aromatic scope.
    """
    from ._descriptors_backend import calc_rdkit

    cls = _scope.cho_class(smiles)
    if cls not in _scope.AROMATIC_SCOPE:
        return Prediction(smiles, None, None, None, None, cho_class=cls,
                          benzoin_relevant=False, method="surrogate_2d",
                          error=f"out_of_scope:{cls} (model is aromatic-aldehyde only)")
    model, feats, medians = _load_surrogate(models_dir)
    bz = _th._make_benzoin_smiles(smiles)

    row = calc_rdkit(smiles)
    if row.get("MW") is None:
        return Prediction(smiles, bz, None, None, None, cho_class=cls,
                          method="surrogate_2d", error="rdkit_parse_failed")
    X = _feat.build_vector(row, None, feats, medians)   # dG_xtb unused (not a 2D feature)
    import pandas as pd
    dG_pred = float(model.predict(pd.DataFrame(X, columns=feats))[0])
    return Prediction(smiles, bz, dG_pred, None, None,
                      favorable=bool(dG_pred < 0),
                      confidence=_confidence("unknown", row.get("RotBonds")),
                      ad_flag="surrogate", cho_class=cls, benzoin_relevant=True,
                      method="surrogate_2d", model_version="surrogate_2d")


def predict_dG_champion(smiles: str, *, xtb_bin: str | None = None,
                        n_confs: int = 10, xtb_cores: int = 2):
    """CHAMPION tier: the real full-library validated model (test MAE 1.503
    kcal/mol on 219k r2SCAN-3c labels), NOT the same artifact as the default
    `predict_dG()` (which uses the older/stale `models/delta_model.joblib` —
    see memory `cross-benzoin-push-20260714` / `gxtb-dft-correction-champion`
    for why the two diverged and why this was shipped separately rather than
    silently replacing the default).

    Slower than `predict_dG()`: computes a full funnel_v3 xTB+morfeus pass on
    both the aldehyde and its self-condensation product (no Multiwfn needed).
    Returns a `_ensemble72_inference.ChampionPrediction` with `dG_pred`,
    `dG_gxtb` baseline, `dG_correction`, a quantile-PI `uncertainty`, and a
    `route_to_dft` flag (the shipped model's own most-uncertain-15% cutoff).
    """
    from . import _ensemble72_inference as _e72
    if not _e72.available():
        from ._ensemble72_inference import ChampionPrediction
        return ChampionPrediction(smiles, None, None, None, None,
                                  error="champion_model_unavailable")
    return _e72.predict_dG_champion(smiles, xtb_bin=xtb_bin, n_confs=n_confs,
                                    xtb_cores=xtb_cores)


def _format(p: Prediction) -> str:
    if not p.benzoin_relevant:
        return (f"{p.smiles}\n  out of scope ({p.cho_class or 'non-aromatic'}) — "
                f"model is aromatic-aldehyde only")
    if p.dG_pred is None:
        return f"{p.smiles}\n  ERROR: {p.error}"
    verdict = "FAVORABLE  (ΔG<0)" if p.favorable else "unfavorable (ΔG>0)"
    if p.method == "surrogate_2d":
        tail = "   [2D surrogate — screening tier, pure SMILES, no xTB]"
    else:
        tail = (f"   [xTB baseline {p.dG_xtb:+.1f}, Δ-correction {p.dG_correction:+.1f}]")
    return (f"{p.smiles}\n"
            f"  ΔG = {p.dG_pred:+6.1f} kcal/mol   {verdict}\n"
            f"  confidence: {p.confidence}   (AD: {p.ad_flag}"
            + (f", dist {p.ad_distance:.2f}" if p.ad_distance is not None else "")
            + ")"
            + (f"\n  model: {p.model_version}" if p.model_version else "")
            + tail)
