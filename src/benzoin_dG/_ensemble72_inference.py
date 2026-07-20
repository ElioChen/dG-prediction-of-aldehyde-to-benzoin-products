"""Inference adapter for the real full-library champion model
(`pipeline/models/gxtb_dft_correction_ENSEMBLE72_20260626.joblib`).

This is the model `pipeline/analysis/finalize_correction.py` actually trained
and validated (test MAE 1.503 kcal/mol on the full 219k-label homo campaign) —
NOT the same artifact as `src/benzoin_dG/models/delta_model.joblib`, which
predates the modern role-aware featurizer and is a materially worse/stale
model (see memory `cross-benzoin-push-20260714`, `gxtb-dft-correction-champion`).

The champion's 72 features (34 product QM + 22 `ald_`-prefixed aldehyde QM +
16 `g_`-prefixed global RDKit-2D on the product SMILES) all come from
`cross_benzoin/cb_featurize.py`'s `featurize_aldehyde`/`featurize_pair` — the
SAME modern per-molecule xTB/morfeus computation used for the 220k-aldehyde
homo_v6 campaign and every cross-benzoin round since. None of these 72
features need Multiwfn (only the separately-tracked, all-null ADCH/QTAIM
columns do), so inference only needs an xTB binary and (for the g-xTB
baseline) the g-xTB binary — no Multiwfn dependency at inference time.

Bridges to `cross_benzoin/` (which itself bridges to `pipeline/compute/`) the
same way `_gxtb_inference.py` bridges to `pipeline/compute/` directly: single
research repo, vendoring the whole funnel_v3+xTB+morfeus stack would be a
large and risky duplication, so we import the validated modules in place.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_CROSS = _REPO / "cross_benzoin"
_COMPUTE = _REPO / "pipeline" / "compute"
for _p in (_COMPUTE, _CROSS):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_MODEL_PATH = _REPO / "pipeline" / "models" / "gxtb_dft_correction_ENSEMBLE72_20260626.joblib"
_PREDICT_SCRIPT = Path(__file__).resolve().parent / "_ensemble72_predict_subprocess.py"
# The bundle was pickled under numpy>=2 (RandomState/BitGenerator objects inside the
# sklearn/xgboost members) and the main project venv (nhc-workflow) ships numpy 1.26.4,
# which cannot unpickle it -- `ValueError: <class 'numpy.random._mt19937.MT19937'> is
# not a known BitGenerator module`. Rather than upgrading the shared venv's numpy (risks
# breaking the rest of the xTB/RDKit pipeline) or vendoring a converted copy of the
# bundle, the scaler/predict/quantile step runs in a short-lived subprocess under
# envs/bde_lite (numpy 2.4.6, has joblib/sklearn/xgboost) -- see
# `_ensemble72_predict_subprocess.py`.
_BDE_LITE_PY = _REPO.parent / "envs" / "bde_lite" / "bin" / "python"

# Must match pipeline/analysis/finalize_correction.py's FEATS construction exactly.
PROD_QM = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta", "xtb_omega",
           "xtb_dipole", "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO", "mulliken_hydH",
           "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb", "fukui_plus_ketC", "fukui_minus_ketC", "dual_ketC",
           "fukui_plus_carbC", "fukui_minus_carbC", "dual_carbC", "vbur_ketC", "vbur_carbC", "sterimol_L",
           "sterimol_B1", "sterimol_B5", "SASA_total", "P_int", "pa_ketO", "hb_dist", "hb_angle", "dih_core"]
ALD = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta", "xtb_omega", "xtb_dipole",
       "mulliken_CHO_C", "mulliken_CHO_O", "fukui_plus_CHO_C", "fukui_minus_CHO_C", "dual_descriptor_CHO_C",
       "wbo_CO", "pa_CHO_O", "vbur_CHO_C", "sterimol_L", "sterimol_B1", "sterimol_B5", "SASA_total", "P_int"]
ALDp = [f"ald_{c}" for c in ALD]
GKEYS = ["TPSA", "HBD", "HBA", "RotB", "FracCsp3", "nHetero", "MolWt", "nRing", "nAromRing", "nAliphRing",
         "nAmide", "has_P", "has_B", "has_S", "has_Si", "has_halogen"]
GLOB = [f"g_{k}" for k in GKEYS]
FEATS = PROD_QM + ALDp + GLOB
HARTREE_TO_KCAL = 627.509474


@dataclass
class ChampionPrediction:
    smiles: str
    benzoin_smiles: str | None
    dG_pred: float | None
    dG_gxtb: float | None
    dG_correction: float | None
    uncertainty: float | None = None    # quantile-PI width (0.95-0.05), same units as ΔG
    route_to_dft: bool | None = None    # True => uncertainty above the shipped routing threshold
    error: str = ""


def _gfeats(smi: str) -> dict:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Descriptors
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return {f"g_{k}": np.nan for k in GKEYS}
    s = {a.GetSymbol() for a in m.GetAtoms()}
    vals = [rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),
            rdMolDescriptors.CalcNumHeteroatoms(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumAliphaticRings(m),
            rdMolDescriptors.CalcNumAmideBonds(m), int("P" in s), int("B" in s), int("S" in s),
            int("Si" in s), int(bool(s & {"F", "Cl", "Br", "I"}))]
    return {f"g_{k}": v for k, v in zip(GKEYS, vals)}


def _predict_via_subprocess(row: dict) -> dict:
    """Runs the scaler/ensemble/quantile step in envs/bde_lite (numpy>=2) --
    see the module docstring's note by `_BDE_LITE_PY` for why this can't run
    in-process under the main venv's numpy 1.26.4."""
    payload = json.dumps({"feature_order": FEATS, "features": row})
    result = subprocess.run([str(_BDE_LITE_PY), str(_PREDICT_SCRIPT)],
                           input=payload, capture_output=True, text=True, timeout=120)
    try:
        out = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": f"subprocess_bad_output: stdout={result.stdout!r} stderr={result.stderr[-500:]!r}"}
    return out


def available() -> bool:
    return _MODEL_PATH.exists() and _CROSS.is_dir() and _BDE_LITE_PY.is_file()


def predict_dG_champion(smiles: str, *, xtb_bin: str | None = None,
                        n_confs: int = 10, xtb_cores: int = 2,
                        timeout: int = 900) -> ChampionPrediction:
    """Predict homo-benzoin ΔG with the real full-library champion (MAE 1.503).

    Computes the aldehyde + self-condensation product on the funnel_v3
    geometry with a full xTB+morfeus pass (no Multiwfn needed), builds the
    72-feature vector exactly as `finalize_correction.py` did at training
    time, and returns the ensemble-mean DFT correction on top of the g-xTB
    baseline ΔG, plus a quantile-PI uncertainty and the shipped route-to-DFT
    flag (most-uncertain ~15% of the training distribution).
    """
    from . import config as _cfg
    import cb_featurize as CF          # noqa: E402 (needs _CROSS on sys.path)
    import thermo_orca as Th           # noqa: E402
    from rdkit import Chem             # noqa: E402

    xtb = _cfg.find_xtb(xtb_bin)
    if not xtb:
        return ChampionPrediction(smiles, None, None, None, None, error="xtb_not_found")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ChampionPrediction(smiles, None, None, None, None, error="invalid_smiles")

    bz = Th._make_benzoin_smiles(smiles)
    if not bz:
        return ChampionPrediction(smiles, None, None, None, None, error="benzoin_smiles_failed")

    tmp = Path(tempfile.mkdtemp(prefix="benzoin_champion_"))
    xyz_dir = tmp / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    try:
        ald_row, g_pair = CF.featurize_aldehyde(
            "query", smiles, xyz_dir=xyz_dir, work_dir=tmp, xtb_bin=xtb, mwf_bin=None,
            do_multiwfn=False, solvent="dmso", n_confs=n_confs, T=298.15, P=1.0,
            cores=xtb_cores, jobs=1, timeout=timeout, conformer="funnel_v3")
        if ald_row.get("error") or g_pair is None:
            return ChampionPrediction(smiles, bz, None, None, None,
                                      error=f"aldehyde_featurize:{ald_row.get('error')}")

        g_cache = {Chem.MolToSmiles(mol, canonical=True): g_pair}
        pair_rec = {"donor_id": "query", "acceptor_id": "query",
                    "donor_smiles": smiles, "acceptor_smiles": smiles}
        prod_row = CF.featurize_pair(
            pair_rec, g_cache=g_cache, xyz_dir=xyz_dir, work_dir=tmp,
            xtb_bin=xtb, mwf_bin=None, do_multiwfn=False, solvent="dmso",
            n_confs=n_confs, T=298.15, P=1.0, cores=xtb_cores, jobs=1,
            timeout=timeout, conformer="funnel_v3")
        if prod_row.get("error") or prod_row.get("dG_gxtb_kcal") is None:
            return ChampionPrediction(smiles, bz, None, None, None,
                                      error=f"product_featurize:{prod_row.get('error')}")

        dG_gxtb = float(prod_row["dG_gxtb_kcal"])
        row = {**{c: prod_row.get(c) for c in PROD_QM},
               **{f"ald_{c}": ald_row.get(c) for c in ALD},
               **_gfeats(prod_row["smiles"])}
        missing = [k for k in FEATS if row.get(k) is None or (isinstance(row.get(k), float) and np.isnan(row[k]))]
        if missing:
            return ChampionPrediction(smiles, bz, None, dG_gxtb, None,
                                      error=f"missing_features:{missing[:5]}"
                                            f"{'...' if len(missing) > 5 else ''}")

        result = _predict_via_subprocess(row)
        if "error" in result:
            return ChampionPrediction(smiles, bz, None, dG_gxtb, None,
                                      error=f"champion_predict:{result['error']}")
        correction = float(result["correction"])
        width = float(result["uncertainty"])
        route = bool(result["route_to_dft"])
        dG_pred = dG_gxtb + correction
        return ChampionPrediction(smiles, bz, dG_pred, dG_gxtb, correction,
                                  uncertainty=width, route_to_dft=route)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
