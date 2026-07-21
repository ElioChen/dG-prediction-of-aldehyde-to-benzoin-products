#!/usr/bin/env python
"""
Paired bootstrap for Phase 2's attentive2d vs distattn GNN-only MAE delta (2.329 vs
2.295, see gnn3d_train_and_compare_v2.py). Same methodology as gnn3d_bootstrap_compare.py
(Phase 1), adapted for coordinate-based graphs and the TripleGNNDistAttn architecture.

Usage
  python cross_benzoin/gnn3d_bootstrap_compare_v2.py \
      --table data/cross_benzoin/cross_round9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet \
      --coords data/cross_benzoin/gnn3d/product_coords_r2348_r9.parquet \
      --champion-dir data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1 \
      --state-2d data/cross_benzoin/gnn3d/attentive2d_v2_matched/models/gnn_state.pt \
      --state-dist data/cross_benzoin/gnn3d/distattn_v1/models/gnn_state.pt \
      --seed 0 --n-boot 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cross_gnn import build_graphs, make_loader, evl, dev  # noqa: E402
from gnn_architectures import TripleGNNAttn, TripleGNNDistAttn  # noqa: E402
from gnn3d_train_and_compare_v2 import build_graphs_dist  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--champion-dir", required=True)
    ap.add_argument("--state-2d", required=True)
    ap.add_argument("--state-dist", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-val-frac", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-dist-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    args = ap.parse_args()

    feats = json.loads((Path(args.champion_dir) / "models" / "feature_list.json").read_text())
    df = pd.read_parquet(args.table)
    co_df = pd.read_parquet(args.coords)
    coords_map = dict(zip(co_df["id"], co_df["coords_flat"]))

    df = df[df["id"].isin(coords_map.keys())].reset_index(drop=True)
    df = df[df["new_scaffold_split"] != "mixed"].reset_index(drop=True)
    df["_split"] = df["new_scaffold_split"]
    rng = np.random.default_rng(args.seed)
    train_pairs_all = df.loc[df["_split"] == "train", "pair_key"].unique()
    n_extra_val = int(len(train_pairs_all) * args.extra_val_frac)
    val_pairs_extra = set(rng.choice(train_pairs_all, size=n_extra_val, replace=False))
    promote = (df["_split"] == "train") & df["pair_key"].isin(val_pairs_extra)
    df.loc[promote, "_split"] = "validation_extra"
    train_mask = (df["_split"] == "train").to_numpy()
    print(f"reconstructed: train={train_mask.sum()} test={(df['_split']=='test').sum()}", flush=True)

    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    med = Xdf[train_mask].median(numeric_only=True)
    Xz = Xdf.fillna(med).fillna(0.0)
    qm_mean = Xz[train_mask].mean().to_numpy()
    qm_std = Xz[train_mask].std().replace(0, 1).to_numpy()
    qm_std = np.where(qm_std == 0, 1.0, qm_std)
    qmz = ((Xz.to_numpy() - qm_mean) / qm_std).astype(np.float32)

    pairs_2d = build_graphs(df, qmz)
    pairs_dist = build_graphs_dist(df, qmz, coords_map)
    id_to_split = dict(zip(df["id"].astype(str), df["_split"]))
    te_2d = [p for p in pairs_2d if id_to_split.get(p.row_id) == "test"]
    te_dist = [p for p in pairs_dist if id_to_split.get(p.row_id) == "test"]
    print(f"test graphs: 2d={len(te_2d)} dist={len(te_dist)}", flush=True)

    ys = torch.tensor([p.y.item() for p in pairs_2d if id_to_split.get(p.row_id) == "train"])
    ym, ysd = ys.mean().item(), ys.std().item()
    for p in pairs_2d + pairs_dist:
        p.ystd = (p.y - ym) / ysd

    ad, bd = te_2d[0].x_p.shape[1], te_2d[0].edge_attr_a.shape[1]
    nqm = len(feats)

    m2d = TripleGNNAttn(ad, bd, nqm, h=args.hidden, layers=args.layers).to(dev)
    m2d.load_state_dict(torch.load(args.state_2d, map_location=dev, weights_only=False))
    mdist = TripleGNNDistAttn(ad, bd, nqm, h=args.hidden, layers=args.layers,
                               n_dist_layers=args.n_dist_layers, n_heads=args.n_heads).to(dev)
    mdist.load_state_dict(torch.load(args.state_dist, map_location=dev, weights_only=False))

    mae2d, yh2d, yt2d, ids2d = evl(m2d, make_loader(te_2d, 64, False), ym, ysd)
    maedist, yhdist, ytdist, idsdist = evl(mdist, make_loader(te_dist, 64, False), ym, ysd)
    print(f"reproduced single-run MAE: 2d={mae2d:.3f} dist={maedist:.3f} "
          f"(should match training-time 2.329 / 2.295)", flush=True)

    by2d = dict(zip(ids2d, zip(yh2d, yt2d)))
    bydist = dict(zip(idsdist, zip(yhdist, ytdist)))
    common = sorted(set(by2d) & set(bydist))
    y_true = np.array([by2d[i][1] for i in common])
    p2 = np.array([by2d[i][0] for i in common])
    pd_ = np.array([bydist[i][0] for i in common])
    ae2 = np.abs(y_true - p2)
    aed = np.abs(y_true - pd_)
    n = len(common)
    print(f"paired test rows: {n}", flush=True)
    obs_delta = aed.mean() - ae2.mean()
    print(f"observed MAE(dist)-MAE(2d) = {obs_delta:+.4f} (negative = dist better)", flush=True)

    rng2 = np.random.default_rng(12345)
    boot_deltas = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng2.integers(0, n, size=n)
        boot_deltas[b] = aed[idx].mean() - ae2[idx].mean()
    p_dist_better = float((boot_deltas < 0).mean())
    ci_lo, ci_hi = np.percentile(boot_deltas, [5, 95])
    print(f"\n=== bootstrap (B={args.n_boot}) ===", flush=True)
    print(f"P(dist better than 2d) = {p_dist_better:.4f}", flush=True)
    print(f"90% CI of delta = ({ci_lo:+.4f}, {ci_hi:+.4f})", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
