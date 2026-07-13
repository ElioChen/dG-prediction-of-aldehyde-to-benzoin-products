#!/usr/bin/env python3
"""
Hybrid D-MPNN (Chemprop v2) core for the benzoin Δ-learning model.

Same Δ-learning setup as the tree path (see delta_core):
    target  y  = dG_orca − dG_xtb          (the DFT correction)
    predict    = dG_xtb + model(graph, x_d)

"Hybrid": the molecular **graph** (aldehyde SMILES) is the message-passing input,
and the **62 descriptors + dG_xtb** (exactly delta_core's feature matrix X) ride
along as Chemprop *extra datapoint descriptors* `x_d`, concatenated into the FFN
readout. So the GNN only has to add graph signal on top of what the trees already
see — the right inductive bias at small n.

Apples-to-apples with the trees: we reuse `delta_core.load_training_table` (same
QC, same X, same target) and the **same** `RepeatedKFold(seed)` splits as
`delta_core.cv_evaluate`, so out-of-fold predictions are directly comparable.

Scaling is leakage-safe: StandardScaler for x_d and for y is fit on the *inner
training* rows of each fold only, never on val/test. chemprop imports are local so
this module imports fine under interpreters without torch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

# delta_core lives one dir up (pipeline/); import it regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import delta_core as dc  # noqa: E402

SMILES_COL = "SMILES"

# Small-data-friendly defaults; sweep_gnn searches around these.
DEFAULT_PARAMS = dict(
    depth=4,             # message-passing steps
    message_hidden=300,  # d_h of the message passing
    ffn_hidden=300,
    ffn_layers=2,
    dropout=0.0,
    batch_size=64,
    max_epochs=120,
    patience=20,         # early-stopping patience on val loss
    ensemble=1,          # models averaged per fold
    val_frac=0.1,        # inner val carved from each outer-train fold for early stop
)


# ─────────────────────────── model construction ───────────────────────────
def build_mpnn(n_xd: int, params: dict):
    """Untrained hybrid MPNN: BondMessagePassing + MeanAggregation + RegressionFFN.

    The predictor's input_dim must include the concatenated x_d (chemprop's MPNN
    cats x_d onto the graph embedding before the FFN).
    """
    from chemprop import models, nn as cpnn

    mp = cpnn.BondMessagePassing(depth=params["depth"], d_h=params["message_hidden"])
    agg = cpnn.MeanAggregation()
    ffn = cpnn.RegressionFFN(
        input_dim=mp.output_dim + n_xd,
        hidden_dim=params["ffn_hidden"],
        n_layers=params["ffn_layers"],
        dropout=params["dropout"],
    )
    return models.MPNN(mp, agg, ffn, batch_norm=True)


def _make_dataset(smiles, X, y, featurizer):
    """MoleculeDataset of (graph, x_d=X-row, y=[corr]). y may be None for predict."""
    from chemprop import data

    dps = []
    for i, smi in enumerate(smiles):
        yi = None if y is None else np.array([y[i]], dtype=float)
        dps.append(data.MoleculeDatapoint.from_smi(smi, y=yi,
                                                   x_d=np.asarray(X[i], dtype=float)))
    return data.MoleculeDataset(dps, featurizer)


def _train_one(tr_dset, val_dset, params, seed):
    from chemprop import data
    from lightning import pytorch as pl

    pl.seed_everything(seed, workers=True)
    model = build_mpnn(tr_dset[0].x_d.shape[0], params)
    tr_loader = data.build_dataloader(tr_dset, batch_size=params["batch_size"],
                                      shuffle=True, num_workers=0)
    val_loader = data.build_dataloader(val_dset, batch_size=params["batch_size"],
                                       shuffle=False, num_workers=0)
    cbs = [pl.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                      patience=params["patience"])]
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"], accelerator="auto", devices=1,
        enable_progress_bar=False, enable_checkpointing=False, logger=False,
        callbacks=cbs, deterministic=False,
    )
    trainer.fit(model, tr_loader, val_loader)
    return model, trainer


def _predict_scaled(model, trainer, dset, params):
    """Predictions in the *scaled-y* space (caller unscales)."""
    from chemprop import data

    loader = data.build_dataloader(dset, batch_size=params["batch_size"],
                                   shuffle=False, num_workers=0)
    preds = trainer.predict(model, loader)
    return np.concatenate([p.numpy() for p in preds]).ravel()


# ─────────────────────────── cross-validation ───────────────────────────
def cv_evaluate_gnn(tbl: dc.TrainTable, params: dict | None = None,
                    folds: int = 5, repeats: int = 3, seed: int = 42,
                    smiles_col: str = SMILES_COL, verbose: bool = True):
    """Repeated K-fold CV of the hybrid D-MPNN, returns (delta, base, oof_pred).

    Mirrors `delta_core.cv_evaluate` exactly (same RepeatedKFold(seed) splits) so
    the GNN's OOF predictions line up fold-for-fold with the tree models'.
    """
    from chemprop import featurizers

    params = {**DEFAULT_PARAMS, **(params or {})}
    smiles = tbl.df[smiles_col].to_numpy()
    Xraw = tbl.X.to_numpy(dtype=float)
    y = tbl.y.astype(float)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    oof_sum = np.zeros(len(y))
    oof_cnt = np.zeros(len(y))

    for fi, (tr, te) in enumerate(rkf.split(Xraw)):
        # Inner val split (for early stopping); deterministic per fold.
        rng = np.random.default_rng(seed + fi)
        perm = rng.permutation(tr)
        n_val = max(1, int(round(params["val_frac"] * len(tr))))
        val_idx, tr_in = perm[:n_val], perm[n_val:]

        # Leakage-safe scaling: fit on inner-train only.
        xs = StandardScaler().fit(Xraw[tr_in])
        ys = StandardScaler().fit(y[tr_in].reshape(-1, 1))
        Xs = xs.transform(Xraw)
        ysc = ys.transform(y.reshape(-1, 1)).ravel()

        tr_dset = _make_dataset(smiles[tr_in], Xs[tr_in], ysc[tr_in], featurizer)
        val_dset = _make_dataset(smiles[val_idx], Xs[val_idx], ysc[val_idx], featurizer)
        te_dset = _make_dataset(smiles[te], Xs[te], None, featurizer)

        # Ensemble: average test predictions over `ensemble` seeds.
        acc = np.zeros(len(te))
        for k in range(params["ensemble"]):
            model, trainer = _train_one(tr_dset, val_dset, params, seed + 100 * fi + k)
            pred_sc = _predict_scaled(model, trainer, te_dset, params)
            acc += ys.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
        oof_sum[te] += acc / params["ensemble"]
        oof_cnt[te] += 1
        if verbose:
            print(f"  fold {fi+1}/{folds*repeats} done "
                  f"(n_tr={len(tr_in)} n_val={len(val_idx)} n_te={len(te)})", flush=True)

    oof = oof_sum / np.maximum(oof_cnt, 1)        # mean correction over repeats
    pred_dft = tbl.dG_xtb + oof
    delta = dc.metrics_vs_dft(tbl.dG_dft, pred_dft)
    base = dc.metrics_vs_dft(tbl.dG_dft, tbl.dG_xtb)
    return delta, base, pred_dft


def fit_full(tbl: dc.TrainTable, params: dict | None = None, seed: int = 42,
             smiles_col: str = SMILES_COL):
    """Fit one (or an ensemble of) model(s) on ALL data; return (models, scalers).

    Used to ship/save the final model after CV selection. Returns the fitted x_d
    and y StandardScalers so inference can reproduce the exact preprocessing.
    """
    from chemprop import featurizers

    params = {**DEFAULT_PARAMS, **(params or {})}
    smiles = tbl.df[smiles_col].to_numpy()
    Xraw = tbl.X.to_numpy(dtype=float)
    y = tbl.y.astype(float)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_val = max(1, int(round(params["val_frac"] * len(y))))
    val_idx, tr_in = perm[:n_val], perm[n_val:]

    xs = StandardScaler().fit(Xraw[tr_in])
    ys = StandardScaler().fit(y[tr_in].reshape(-1, 1))
    Xs = xs.transform(Xraw)
    ysc = ys.transform(y.reshape(-1, 1)).ravel()

    tr_dset = _make_dataset(smiles[tr_in], Xs[tr_in], ysc[tr_in], featurizer)
    val_dset = _make_dataset(smiles[val_idx], Xs[val_idx], ysc[val_idx], featurizer)

    models_out = []
    for k in range(params["ensemble"]):
        model, _ = _train_one(tr_dset, val_dset, params, seed + k)
        models_out.append(model)
    return models_out, {"x_scaler": xs, "y_scaler": ys, "params": params}
