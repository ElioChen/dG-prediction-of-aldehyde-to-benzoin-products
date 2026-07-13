#!/usr/bin/env python3
"""
Shared core for the benzoin Δ-learning model (used by train_delta.py and
sweep_delta.py, and reusable by the later RDKit-2D surrogate).

Δ-learning setup
    target  y   = dG_target − dG_xtb_kcal            (the DFT correction)
    features X  = ~62 descriptors + dG_xtb_kcal
    predict     = dG_xtb_kcal + model(X)  ≈ DFT-level ΔG

Everything here is side-effect free: load the table, build a model, run
(repeated) K-fold CV and get out-of-fold predictions + metrics. MLflow logging
and CLI live in the callers.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

# MLflow 3.x put the plain file store in maintenance mode and errors on it, so we
# use a SQLite tracking backend (full-featured) + a file artifact root. The UI is
# launched the same way:  mlflow ui --backend-store-uri <TRACKING_URI>
# Paths are anchored to the repo root (parent of pipeline/) so they're independent
# of the current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{REPO_ROOT}/mlflow.db"     # sqlite:///<abs> (3 slashes + abs)
ARTIFACT_URI = f"file:{REPO_ROOT}/mlartifacts"
EXPERIMENT = "benzoin_delta_dG"


def setup_mlflow(experiment: str | None = None):
    """Point MLflow at the SQLite store and ensure the experiment exists with a
    fixed artifact root (so runs land in the same place regardless of cwd).

    Experiment name resolves from arg -> $MLFLOW_EXPERIMENT -> default, so each
    dataset version can use its own experiment and keep comparisons uncontaminated.
    """
    import os
    import mlflow
    from mlflow.tracking import MlflowClient
    experiment = experiment or os.environ.get("MLFLOW_EXPERIMENT") or EXPERIMENT
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        client.create_experiment(experiment, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(experiment)

# Non-feature columns in the descriptor table.
ID_COLS = ["index", "SMILES", "PubChem_CID", "xtb_optimized", "error", "xyz_file"]
FILTER_COLS = ["n_CHO"]
XTB_DG = "dG_xtb_kcal"
# Two DFT-level targets are available from thermo_orca's delta_G.csv:
#   dG_orca_kcal         ORCA r2SCAN-3c SP (CPCM-DMSO) + xTB-RRHO thermal corrections
#   dG_orca_shermo_kcal  ORCA r2SCAN-3c SP (CPCM-DMSO) + Shermo thermal corrections
# NB: r2SCAN-3c is the production DFT level; PBE0-D4 is DEPRECATED (data/labels_pbe0_old).
DEFAULT_TARGET = "dG_orca_kcal"
LABEL_EXTRA = ["dG_orca_kcal", "dG_orca_shermo_kcal", "dG_shermo_kcal"]


@dataclass
class TrainTable:
    df: pd.DataFrame                 # joined, cleaned rows
    feats: list[str]                 # feature column names (incl. dG_xtb_kcal)
    target: str                      # DFT target column name
    X: pd.DataFrame                  # feature matrix (median-imputed)
    y: np.ndarray                    # correction = dG_target − dG_xtb
    dG_xtb: np.ndarray
    dG_dft: np.ndarray
    medians: dict[str, float] = field(default_factory=dict)


def _read_glob(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  {pattern}: {len(files)} files -> {len(df):,} rows")
    return df


# Single source of truth: the assembled unified-featurize table (r2SCAN-3c,
# descriptors + dG_xtb + dG_orca on ONE geometry). Built by assemble_featurize.py.
DEFAULT_FEATURIZE_PARQUET = str(REPO_ROOT / "data/featurize.parquet")
# Legacy chunk-CSV path (separate descriptors + labels) — fallback only.
DEFAULT_DESC_GLOB = str(REPO_ROOT / "data/descriptors/chunk_*/descriptors.csv")
DEFAULT_LABEL_GLOB = str(REPO_ROOT / "data/labels/chunk_*/delta_G.csv")

# Columns that are text / identifiers, not numeric features.
_STR_COLS = ("SMILES", "aldehyde_smiles", "aldehyde_name", "benzoin_smiles",
             "error", "xyz_file", "PubChem_CID")


# Reactive / exotic SMILES classes excluded after the n=1500 energy diagnostic
# (they carried 2–3× the correction noise). filter_smiles.classify is the single
# source of truth for these rules.
_REACTIVE_REASONS = {"isotope", "zwitterion_or_nitro", "reactive_group"}

# PRODUCTION SCOPE: the benzoin condensation works on AROMATIC aldehydes; aliphatic
# substrates (α-H → enolization/aldol) give poor yields and are off-target. Per the
# CHO-environment categorization, training/prediction are restricted to CHO bonded
# to a carbo- or hetero-aromatic ring. Dropping aliphatic costs only ~0.07 kcal on
# aromatic CV (a single global model still wins over a class-split one). Set to None
# to train on all categories. cho_category.cho_class is the single source of truth.
CHO_SCOPE = {"aromatic_carbo", "aromatic_hetero"}


def _finalize_table(full: pd.DataFrame, target: str, qc: bool,
                    mag_max: float, k_mad: float,
                    drop_reactive: bool = True,
                    cho_scope: set | None = CHO_SCOPE) -> TrainTable:
    """QC + feature-matrix build, shared by the parquet and chunk-glob loaders.

    `full` already carries the descriptor columns AND `dG_xtb_kcal` + `target`.
    QC targets *calculation failures*, not unfavourable reactions: we drop rows
    whose |ΔG| is unphysical (> `mag_max`) or whose correction is a robust outlier
    (|corr − median| > `k_mad`·MAD) — the signature of a bad conformer / non-
    converged xTB. `drop_reactive` removes exotic/reactive SMILES classes (nitro,
    nitroso, azo, isotopes, ...) per filter_smiles.classify; `cho_scope` restricts
    to the aromatic-aldehyde production scope (carbo + hetero), dropping aliphatic.
    """
    full = full.copy()
    if drop_reactive and "SMILES" in full.columns:
        try:
            from filter_smiles import classify
            bad = full["SMILES"].map(classify).isin(_REACTIVE_REASONS)
            if bad.any():
                print(f"  reactive/exotic filter: dropped {int(bad.sum())} rows "
                      f"(nitro/nitroso/azo/isotope/zwitterion)")
                full = full[~bad]
        except Exception as e:
            print(f"  (reactive filter skipped: {e})")
    if cho_scope and "SMILES" in full.columns:
        try:
            from cho_category import cho_class
            cls = full["SMILES"].map(cho_class)
            keep = cls.isin(cho_scope)
            print(f"  CHO-scope filter (aromatic-only: {sorted(cho_scope)}): "
                  f"kept {int(keep.sum())}/{len(full)} "
                  f"(dropped {int((~keep).sum())} aliphatic/vinyl/other)")
            full = full[keep]
        except Exception as e:
            print(f"  (CHO-scope filter skipped: {e})")
    for c in full.columns:
        if c not in _STR_COLS:
            full[c] = pd.to_numeric(full[c], errors="coerce")
    if "error" in full.columns:
        full = full[full["error"].astype("string").fillna("") == ""]
    if "n_CHO" in full.columns:
        full = full[full["n_CHO"] == 1]
    full = full[full[target].notna() & full[XTB_DG].notna()].copy()
    if qc:
        n0 = len(full)
        corr = full[target] - full[XTB_DG]
        med = corr.median()
        mad = (corr - med).abs().median() or 1.0
        keep = ((full[XTB_DG].abs() <= mag_max) & (full[target].abs() <= mag_max)
                & ((corr - med).abs() <= k_mad * mad))
        full = full[keep].copy()
        print(f"  QC (failure filter: |ΔG|<={mag_max}, |corr−med|<={k_mad}·MAD"
              f"={k_mad*mad:.1f}): kept {len(full)}/{n0} rows")

    drop = set(ID_COLS) | set(FILTER_COLS) | {XTB_DG, target} | set(LABEL_EXTRA)
    feats = [c for c in full.columns
             if c not in drop and pd.api.types.is_numeric_dtype(full[c])]
    feats = feats + [XTB_DG]                       # xTB ΔG is itself a feature

    X = full[feats].copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    y = (full[target] - full[XTB_DG]).to_numpy()
    return TrainTable(
        df=full, feats=feats, target=target, X=X, y=y,
        dG_xtb=full[XTB_DG].to_numpy(), dG_dft=full[target].to_numpy(),
        medians={k: float(v) for k, v in medians.items()},
    )


def load_training_table(
    desc_glob: str = DEFAULT_DESC_GLOB,
    label_glob: str = DEFAULT_LABEL_GLOB,
    target: str = DEFAULT_TARGET,
    qc: bool = True,
    mag_max: float = 45.0,
    k_mad: float = 6.0,
    parquet: str | None = DEFAULT_FEATURIZE_PARQUET,
) -> TrainTable:
    """Model-ready training table.

    Prefers the assembled unified-featurize parquet (`parquet`, descriptors +
    labels on one geometry) when present; otherwise falls back to the legacy
    chunk-CSV path (separate descriptors + labels merged on `index`). Both feed the
    same QC + feature-build in `_finalize_table`.
    """
    if parquet and Path(parquet).exists():
        full = pd.read_parquet(parquet)
        print(f"  {parquet}: {len(full):,} rows (unified featurize)")
        return _finalize_table(full, target, qc, mag_max, k_mad)

    desc = _read_glob(desc_glob)
    lab = _read_glob(label_glob)
    for c in (XTB_DG, target, "index"):
        if c in lab.columns:
            lab[c] = pd.to_numeric(lab[c], errors="coerce")
    keep_lab = ["index", XTB_DG, target] + [c for c in LABEL_EXTRA if c in lab.columns]
    full = desc.merge(lab[keep_lab], on="index", how="inner")
    return _finalize_table(full, target, qc, mag_max, k_mad)


def build_model(kind: str, params: dict | None = None, seed: int = 42):
    """Construct an unfitted regressor. `params` overrides the defaults."""
    params = dict(params or {})
    if kind == "xgb":
        from xgboost import XGBRegressor
        base = dict(n_estimators=600, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                    random_state=seed, n_jobs=-1)
        base.update(params)
        return XGBRegressor(**base)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        base = dict(n_estimators=500, random_state=seed, n_jobs=-1)
        base.update(params)
        return RandomForestRegressor(**base)
    from sklearn.ensemble import GradientBoostingRegressor
    base = dict(random_state=seed)
    base.update(params)
    return GradientBoostingRegressor(**base)


def metrics_vs_dft(dG_dft: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return dict(
        MAE=float(mean_absolute_error(dG_dft, pred)),
        RMSE=float(np.sqrt(mean_squared_error(dG_dft, pred))),
        R2=float(r2_score(dG_dft, pred)),
    )


def cv_evaluate(tbl: TrainTable, kind: str, params: dict | None = None,
                folds: int = 5, repeats: int = 3, seed: int = 42):
    """Repeated K-fold CV. Returns (delta_metrics, baseline_metrics, oof_pred).

    Repeated K-fold averages OOF predictions over `repeats` shuffles to damp the
    split-noise that makes a single 5-fold MAE unreliable at n≈200 — essential so
    model comparison/optuna doesn't chase CV-split luck.
    """
    X, y = tbl.X, tbl.y
    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    oof_sum = np.zeros(len(X))
    oof_cnt = np.zeros(len(X))
    for tr, te in rkf.split(X):
        m = build_model(kind, params, seed)
        m.fit(X.iloc[tr], y[tr])
        oof_sum[te] += m.predict(X.iloc[te])
        oof_cnt[te] += 1
    oof = oof_sum / np.maximum(oof_cnt, 1)
    pred_dft = tbl.dG_xtb + oof
    delta = metrics_vs_dft(tbl.dG_dft, pred_dft)
    base = metrics_vs_dft(tbl.dG_dft, tbl.dG_xtb)     # correction ≡ 0
    return delta, base, pred_dft
