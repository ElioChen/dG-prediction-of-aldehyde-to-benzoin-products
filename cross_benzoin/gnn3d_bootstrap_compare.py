#!/usr/bin/env python
"""
Paired bootstrap significance check for the attentive2d-vs-attentive3d GNN-only MAE delta
found by gnn3d_train_and_compare.py (2.460 vs 2.386 on the geometry-matched test subset,
n=188 -- suggestive but not yet confirmed by this project's own standard, see memory/
session discussion 2026-07-21). Reloads both saved checkpoints (gnn_state.pt), rebuilds
the IDENTICAL restricted dataframe + split + graphs (deterministic given seed=0, same
table/bond-length cache), gets per-row test predictions from both models, then bootstraps
the paired MAE delta (B=20000, same methodology as verify_and_bootstrap_9rounds.py).

Usage
  python cross_benzoin/gnn3d_bootstrap_compare.py \
      --table data/cross_benzoin/cross_round9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet \
      --bond-lengths data/cross_benzoin/gnn3d/product_bond_lengths_r2348_r9.parquet \
      --champion-dir data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1 \
      --state-2d data/cross_benzoin/gnn3d/attentive2d_matched_v1/models/gnn_state.pt \
      --state-3d data/cross_benzoin/gnn3d/attentive3d_v1/models/gnn_state.pt \
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
from gnn_architectures import TripleGNNAttn, TripleGNN3D  # noqa: E402
from gnn3d_train_and_compare import build_graphs_3d  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--bond-lengths", required=True)
    ap.add_argument("--champion-dir", required=True)
    ap.add_argument("--state-2d", required=True)
    ap.add_argument("--state-3d", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-val-frac", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    feats = json.loads((Path(args.champion_dir) / "models" / "feature_list.json").read_text())
    df = pd.read_parquet(args.table)
    bl_df = pd.read_parquet(args.bond_lengths)
    bond_len_map = dict(zip(bl_df["id"], bl_df["bond_lengths"]))

    # ── identical restriction + split reconstruction as gnn3d_train_and_compare.py ──
    df = df[df["id"].isin(bond_len_map.keys())].reset_index(drop=True)
    df = df[df["new_scaffold_split"] != "mixed"].reset_index(drop=True)
    df["_split"] = df["new_scaffold_split"]
    rng = np.random.default_rng(args.seed)
    train_pairs_all = df.loc[df["_split"] == "train", "pair_key"].unique()
    n_extra_val = int(len(train_pairs_all) * args.extra_val_frac)
    val_pairs_extra = set(rng.choice(train_pairs_all, size=n_extra_val, replace=False))
    promote = (df["_split"] == "train") & df["pair_key"].isin(val_pairs_extra)
    df.loc[promote, "_split"] = "validation_extra"
    train_mask = (df["_split"] == "train").to_numpy()
    test_mask = (df["_split"] == "test").to_numpy()
    print(f"reconstructed: train={train_mask.sum()} test={test_mask.sum()} "
          f"(should match n_train=25294 n_test=188 from the training run)", flush=True)

    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    med = Xdf[train_mask].median(numeric_only=True)
    Xz = Xdf.fillna(med).fillna(0.0)
    qm_mean = Xz[train_mask].mean().to_numpy()
    qm_std = Xz[train_mask].std().replace(0, 1).to_numpy()
    qm_std = np.where(qm_std == 0, 1.0, qm_std)
    qmz = ((Xz.to_numpy() - qm_mean) / qm_std).astype(np.float32)

    train_ids = set(df.loc[train_mask, "id"].astype(str))
    train_lens = [d for rid in train_ids for d in bond_len_map.get(rid, [])]
    dist_mean, dist_std = float(np.mean(train_lens)), float(np.std(train_lens))
    dist_std = dist_std if dist_std > 1e-6 else 1.0

    pairs_2d = build_graphs(df, qmz)
    pairs_3d = build_graphs_3d(df, qmz, bond_len_map, dist_mean, dist_std)
    id_to_split = dict(zip(df["id"].astype(str), df["_split"]))
    te_2d = [p for p in pairs_2d if id_to_split.get(p.row_id) == "test"]
    te_3d = [p for p in pairs_3d if id_to_split.get(p.row_id) == "test"]
    print(f"test graphs: 2d={len(te_2d)} 3d={len(te_3d)}", flush=True)

    ys = torch.tensor([p.y.item() for p in pairs_2d if id_to_split.get(p.row_id) == "train"])
    ym, ysd = ys.mean().item(), ys.std().item()
    for p in pairs_2d + pairs_3d:
        p.ystd = (p.y - ym) / ysd

    ad, bd = te_2d[0].x_p.shape[1], te_2d[0].edge_attr_a.shape[1]
    bd_prod3d = te_3d[0].edge_attr_p.shape[1]
    nqm = len(feats)

    m2d = TripleGNNAttn(ad, bd, nqm, h=args.hidden, layers=args.layers).to(dev)
    m2d.load_state_dict(torch.load(args.state_2d, map_location=dev, weights_only=False))
    m3d = TripleGNN3D(ad, bd, bd_prod3d, nqm, h=args.hidden, layers=args.layers).to(dev)
    m3d.load_state_dict(torch.load(args.state_3d, map_location=dev, weights_only=False))

    mae2d, yh2d, yt2d, ids2d = evl(m2d, make_loader(te_2d, 64, False), ym, ysd)
    mae3d, yh3d, yt3d, ids3d = evl(m3d, make_loader(te_3d, 64, False), ym, ysd)
    print(f"reproduced single-run MAE: 2d={mae2d:.3f} 3d={mae3d:.3f} "
          f"(should match training-time 2.460 / 2.386)", flush=True)

    by2d = dict(zip(ids2d, zip(yh2d, yt2d)))
    by3d = dict(zip(ids3d, zip(yh3d, yt3d)))
    common = sorted(set(by2d) & set(by3d))
    y_true = np.array([by2d[i][1] for i in common])
    p2 = np.array([by2d[i][0] for i in common])
    p3 = np.array([by3d[i][0] for i in common])
    ae2 = np.abs(y_true - p2)
    ae3 = np.abs(y_true - p3)
    n = len(common)
    print(f"paired test rows: {n}", flush=True)
    obs_delta = ae3.mean() - ae2.mean()
    print(f"observed MAE(3d)-MAE(2d) = {obs_delta:+.4f} (negative = 3d better)", flush=True)

    rng2 = np.random.default_rng(12345)
    boot_deltas = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng2.integers(0, n, size=n)
        boot_deltas[b] = ae3[idx].mean() - ae2[idx].mean()
    p_3d_better = float((boot_deltas < 0).mean())
    ci_lo, ci_hi = np.percentile(boot_deltas, [5, 95])
    print(f"\n=== bootstrap (B={args.n_boot}) ===", flush=True)
    print(f"P(3d better than 2d) = {p_3d_better:.4f}", flush=True)
    print(f"90% CI of delta = ({ci_lo:+.4f}, {ci_hi:+.4f})", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
