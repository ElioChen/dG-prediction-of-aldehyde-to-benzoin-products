"""Applicability domain (AD) — is a query molecule inside the model's experience?

The Δ-learning model only saw ~440 aldehydes out of a vast, diffuse chemical
space (median nearest-training Tanimoto distance ≈ 0.66), so most new molecules
are extrapolation. This flags each prediction by the standardized **descriptor-
space** distance to its nearest training molecule — distance in the features the
model actually uses, not fingerprints.

Reference data (training feature stats + per-feature scale + the training
nearest-neighbour distance distribution) ships as `models/ad_reference.npz`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MODELS = Path(__file__).resolve().parent / "models"


@dataclass
class ADResult:
    distance: float      # standardized-Euclidean distance to nearest training molecule
    flag: str            # "in_domain" | "borderline" | "extrapolation"
    percentile: float    # where `distance` falls in the training NN-distance distribution


class ADReference:
    def __init__(self, means, stds, X_std, train_nn, p_in, p_edge):
        self.means = np.asarray(means, float)
        self.stds = np.asarray(stds, float)
        self.X_std = np.asarray(X_std, float)        # standardized training matrix
        self.train_nn = np.asarray(train_nn, float)  # training LOO nearest-neighbour dists
        self.p_in = float(p_in)                      # in-domain threshold (e.g. 90th pct)
        self.p_edge = float(p_edge)                  # edge threshold (e.g. 99th pct)

    @classmethod
    def load(cls, models_dir=None) -> "ADReference | None":
        path = Path(models_dir or _MODELS) / "ad_reference.npz"
        if not path.exists():
            return None
        d = np.load(path)
        return cls(d["means"], d["stds"], d["X_std"], d["train_nn"],
                   float(d["p_in"]), float(d["p_edge"]))

    def score(self, raw_vector: np.ndarray) -> ADResult:
        q = (np.asarray(raw_vector, float).ravel() - self.means) / self.stds
        dist = float(np.sqrt(((self.X_std - q) ** 2).sum(axis=1)).min())
        pct = float((self.train_nn < dist).mean() * 100.0)
        if dist <= self.p_in:
            flag = "in_domain"
        elif dist <= self.p_edge:
            flag = "borderline"
        else:
            flag = "extrapolation"
        return ADResult(distance=dist, flag=flag, percentile=pct)


def build_reference(X_train: np.ndarray, p_in_q: float = 0.90,
                    p_edge_q: float = 0.99) -> dict:
    """Compute the npz payload from the training feature matrix (raw, imputed)."""
    X = np.asarray(X_train, float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds
    # leave-one-out nearest-neighbour distance per training point
    nn = np.empty(len(X_std))
    for i in range(len(X_std)):
        d = np.sqrt(((X_std - X_std[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        nn[i] = d.min()
    return dict(means=means, stds=stds, X_std=X_std, train_nn=nn,
                p_in=float(np.quantile(nn, p_in_q)),
                p_edge=float(np.quantile(nn, p_edge_q)))
