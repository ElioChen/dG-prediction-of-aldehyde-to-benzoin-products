#!/usr/bin/env python
"""
Single deployable inference artifact for the cross-benzoin Delta-model:
blends the MLP+XGB tabular ensemble with the triple-encoder GNN (donor +
acceptor + product graphs), both trained on the corrected scaffold-disjoint
split (see memory cross-scaffold-disjoint-rebuild-20260717.md,
cross-round6-ensemble-holds-gnn-stack-null-20260716.md).

Must run under `envs/gnn_lite` (has torch, torch_geometric, sklearn, xgboost,
rdkit all in one place -- verified the ensemble .joblib unpickles cleanly
there too, so unlike homo's ENSEMBLE72 this does NOT need a cross-environment
subprocess bridge; everything runs in a single process/environment).

Input: a DataFrame with the champion's 260 named QM/mordred feature columns
(see models/feature_list.json), plus `donor_smiles`, `acceptor_smiles`,
`smiles` (product) and `dG_gxtb_kcal` (g-xTB baseline) -- i.e. the standard
output of cross_benzoin/cb_featurize.py + assemble_cross_training_table_v3.py,
NOT raw unfeaturized SMILES (mirrors how every other model in this project is
actually invoked -- see score_round_active_learning.py for the equivalent
tabular-only pattern).

Usage (as a library):
    from predict_cross_champion import CrossBenzoinBlendPredictor
    pred = CrossBenzoinBlendPredictor.load("data/cross_benzoin/cross_round7/scaffold_disjoint_v1")
    dG_pred = pred.predict(df)   # np.ndarray, kcal/mol

Usage (CLI smoke test):
    python cross_benzoin/predict_cross_champion.py --table <featurized.parquet> --n 5
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import __main__ as _main_mod  # noqa: E402
from train_cross_ensemble import MLPXGBEnsemble  # noqa: E402
_main_mod.MLPXGBEnsemble = MLPXGBEnsemble

from train_cross_gnn import TripleGNN, graph, TripleData, dev  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

BASELINE_COL = "dG_gxtb_kcal"

# Validated on the scaffold-disjoint holdout (n=450 after dropping 'mixed' rows,
# see train_gnn_scaffold_disjoint_v1/models/metadata.json for the exact sweep) --
# filled in once that job's real number is known; DO NOT trust the old
# candidates_v3-leaky-split w_gnn=0.8/0.95 numbers from train_gnn_retune_b_v1/
# (see memory: that split systematically inflates apparent accuracy).
DEFAULT_BLEND_W_GNN = None  # noqa: set by CrossBenzoinBlendPredictor.load() from metadata.json


@dataclass
class CrossBenzoinBlendPredictor:
    ensemble: MLPXGBEnsemble
    gnn: TripleGNN
    qm_mean: np.ndarray
    qm_std: np.ndarray
    med: pd.Series
    ym: float
    ysd: float
    feats: list[str]
    blend_w_gnn: float

    @classmethod
    def load(cls, model_dir: str | Path, blend_w_gnn: float | None = None,
              gnn_dir: str | Path | None = None) -> "CrossBenzoinBlendPredictor":
        model_dir = Path(model_dir)
        ensemble = joblib.load(model_dir / "models" / "ensemble_scaffold_disjoint.joblib")

        # gnn_dir defaults to the original 80/10/10-split convention for back-compat;
        # pass explicitly for any other split (e.g. train_gnn_scaffold_disjoint_721_v1).
        gnn_dir = Path(gnn_dir) if gnn_dir is not None else model_dir.parent / "train_gnn_scaffold_disjoint_v1"
        stats = joblib.load(gnn_dir / "models" / "gnn_norm_stats.joblib")
        gnn = TripleGNN(stats["ad"], stats["bd"], stats["nqm"],
                         h=stats["hidden"], layers=stats["layers"]).to(dev)
        gnn.load_state_dict(torch.load(gnn_dir / "models" / "cross_gnn_state.pt",
                                        map_location=dev, weights_only=False))
        gnn.eval()

        if blend_w_gnn is None:
            meta = json.loads((gnn_dir / "models" / "metadata.json").read_text())
            blend_w_gnn = meta["best_blend_w_gnn"]
            print(f"[CrossBenzoinBlendPredictor] using validated blend_w_gnn={blend_w_gnn} "
                  f"from {gnn_dir}/models/metadata.json (best_blend_mae={meta['best_blend_mae']:.3f})")

        return cls(ensemble=ensemble, gnn=gnn, qm_mean=stats["qm_mean"], qm_std=stats["qm_std"],
                   med=stats["med"], ym=stats["ym"], ysd=stats["ysd"], feats=stats["feats"],
                   blend_w_gnn=blend_w_gnn)

    def _gnn_predict(self, df: pd.DataFrame) -> np.ndarray:
        Xdf = df[self.feats].apply(pd.to_numeric, errors="coerce")
        Xz = Xdf.fillna(self.med).fillna(0.0)
        qm_std = np.where(self.qm_std == 0, 1.0, self.qm_std)
        qmz = ((Xz.to_numpy() - self.qm_mean) / qm_std).astype(np.float32)

        pairs = []
        keep_idx = []
        for k in range(len(df)):
            gp = graph(df["smiles"].iloc[k])
            gd = graph(df["donor_smiles"].iloc[k])
            ga = graph(df["acceptor_smiles"].iloc[k])
            if gp is None or gd is None or ga is None:
                continue
            d = TripleData()
            d.x_p, d.edge_index_p, d.edge_attr_p = gp
            d.x_d, d.edge_index_d, d.edge_attr_d = gd
            d.x_a, d.edge_index_a, d.edge_attr_a = ga
            d.qm = torch.tensor(qmz[k], dtype=torch.float).view(1, -1)
            pairs.append(d)
            keep_idx.append(k)
        if not pairs:
            return np.full(len(df), np.nan)

        loader = DataLoader(pairs, batch_size=64, shuffle=False, follow_batch=["x_p", "x_d", "x_a"])
        preds = np.full(len(df), np.nan)
        pos = 0
        with torch.no_grad():
            for b in loader:
                b = b.to(dev)
                o = self.gnn(b).cpu().numpy()
                n = o.shape[0]
                preds[[keep_idx[i] for i in range(pos, pos + n)]] = o * self.ysd + self.ym
                pos += n
        return preds

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns predicted dG_orca (kcal/mol), NOT the raw delta -- baseline already added."""
        base = df[BASELINE_COL].to_numpy()
        ens_delta = self.ensemble.predict(df)
        gnn_delta = self._gnn_predict(df)
        blend_delta = (1 - self.blend_w_gnn) * ens_delta + self.blend_w_gnn * gnn_delta
        # if the GNN failed to build a graph for some row (invalid SMILES), fall back to ensemble-only
        nan_mask = np.isnan(gnn_delta)
        if nan_mask.any():
            blend_delta = np.where(nan_mask, ens_delta, blend_delta)
        return base + blend_delta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--model-dir", default="data/cross_benzoin/cross_round7/scaffold_disjoint_v1")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_parquet(args.table).head(args.n)
    predictor = CrossBenzoinBlendPredictor.load(args.model_dir)
    pred = predictor.predict(df)
    for i, (p, actual) in enumerate(zip(pred, df.get("dG_orca_kcal", [None] * len(df)))):
        print(f"row {i}: pred={p:.3f}" + (f"  actual={actual:.3f}  err={abs(p-actual):.3f}" if actual is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
