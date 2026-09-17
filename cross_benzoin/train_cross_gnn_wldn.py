#!/usr/bin/env python
"""WLDN-style (Weisfeiler-Lehman Difference Network, Jin et al. 2017) reaction
comparison point for cross-benzoin's dG GNN leg, user-requested 2026-09-17
("尝试其他的基于反应的GNN") as a third strategy alongside the deployed champion
TripleGNN (train_cross_gnn.py) and the CRG/CGR-delta condensed-graph scripts
(train_cross_gnn_crg.py, train_cross_gnn_cgr_delta.py).

The three dG-GNN strategies this project now has, contrasted:
  TripleGNN   -- THREE separately-weighted encoders (encP/encD/encA), pooled
                 embeddings CONCATENATED at readout. Donor/acceptor atoms can
                 never influence each other's learned embedding (no shared
                 graph, no shared weights).
  CRG/CGR-Δ   -- ONE encoder over ONE connected graph (donor+acceptor atoms
                 physically merged into the product's own graph, reaction-core
                 atoms/bonds explicitly tagged). Message passing DOES let the
                 two sides interact, at the cost of building a bespoke merged
                 graph per row.
  WLDN (here) -- ONE SHARED-weight encoder applied to product/donor/acceptor
                 graphs SEPARATELY (same weights, so all three project into a
                 common embedding space -- unlike TripleGNN's independent
                 per-role weights), combined via the WLDN difference
                 h_product - (h_donor + h_acceptor) instead of concatenation
                 or graph merging. The difference vector is a direct read of
                 "what the reaction changed", the core idea behind Weisfeiler-
                 Lehman Difference Networks for reaction-outcome/property
                 prediction -- no bespoke graph construction needed (reuses
                 the plain per-molecule graphs TripleGNN already builds), but
                 (unlike CRG) message passing still cannot let donor and
                 acceptor atoms interact directly; only their pooled
                 embeddings interact, via subtraction.

Reuses graph()/af()/bf() (atom/bond featurization) and Enc() (GINEConv encoder
block) verbatim from train_cross_gnn.py -- same atom/bond feature set and same
per-encoder capacity as the champion TripleGNN and CRG/CGR-delta, so any MAE
difference across all four scripts is attributable to combination strategy
(concat vs graph-merge vs difference) and weight-sharing, not a featurization
change.

Same target, x_d schema, and `new_scaffold_split`-based split as
train_cross_gnn_crg.py.

Env: /home/schen3/venv/nequip (torch 2.11 + PyTorch Geometric 2.7, the SAME env
the deployed champion TripleGNN and CRG/CGR-delta all train in).

Usage
  # CPU correctness smoke test:
  /home/schen3/venv/nequip/bin/python cross_benzoin/train_cross_gnn_wldn.py \
      --table data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet \
      --feature-list data/cross_benzoin/feature_list_257_no_nCHO_v2.json \
      --outdir /tmp/wldn_smoke --smoke

  # real run (GPU, via sbatch submit_cross_gnn_wldn.sh):
  CB_TARGET_COL=dG_r2scan_kcal CB_BASELINE_COL=dG_b973c_kcal \
  /home/schen3/venv/nequip/bin/python cross_benzoin/train_cross_gnn_wldn.py \
      --table ... --feature-list ... --outdir data/cross_benzoin/cross_round10/gnn_wldn_10rounds_b973c_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cross_gnn import graph  # noqa: E402 -- verbatim atom/bond featurization

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Enc(nn.Module):
    def __init__(self, ad, bd, h=128, layers=4):
        super().__init__()
        self.ne = nn.Linear(ad, h)
        self.ee = nn.Linear(bd, h)
        self.cv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(layers):
            self.cv.append(GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=h))
            self.bn.append(nn.BatchNorm1d(h))

    def forward(self, x, ei, ea, batch):
        x = self.ne(x)
        e = self.ee(ea)
        for c, b in zip(self.cv, self.bn):
            x = x + F.relu(b(c(x, ei, e)))
        return torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], 1)


class TripleData(Data):
    """Same multi-graph-per-example container as train_cross_gnn.py's TripleData
    (product/donor/acceptor each need their own edge_index offset bookkeeping)."""

    def __inc__(self, key, value, *a, **k):
        if key == "edge_index_p":
            return self.x_p.size(0)
        if key == "edge_index_d":
            return self.x_d.size(0)
        if key == "edge_index_a":
            return self.x_a.size(0)
        return super().__inc__(key, value, *a, **k)


class WLDNGNN(nn.Module):
    """SHARED single encoder over product/donor/acceptor graphs (same weights --
    all three land in a common embedding space) + QM-scalar channel, combined via
    the WLDN-style reaction-difference vector h_P - (h_D + h_A) instead of
    concatenation (TripleGNN) or graph merging (CRG/CGR-delta)."""

    def __init__(self, ad, bd, nqm, h=128, layers=4, dropout=0.1):
        super().__init__()
        self.enc = Enc(ad, bd, h, layers)
        din = 2 * h + nqm
        self.head = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, b):
        hP = self.enc(b.x_p, b.edge_index_p, b.edge_attr_p, b.x_p_batch)
        hD = self.enc(b.x_d, b.edge_index_d, b.edge_attr_d, b.x_d_batch)
        hA = self.enc(b.x_a, b.edge_index_a, b.edge_attr_a, b.x_a_batch)
        hdiff = hP - (hD + hA)
        h = torch.cat([hdiff, b.qm.view(-1, b.qm.shape[-1])], 1)
        return self.head(h).squeeze(-1)


def build_graphs(df: pd.DataFrame, qmz: np.ndarray):
    out = []
    n_fail = 0
    for k in range(len(df)):
        gp = graph(df["smiles"].iloc[k])
        gd = graph(df["donor_smiles"].iloc[k])
        ga = graph(df["acceptor_smiles"].iloc[k])
        if gp is None or gd is None or ga is None:
            n_fail += 1
            continue
        d = TripleData()
        d.x_p, d.edge_index_p, d.edge_attr_p = gp
        d.x_d, d.edge_index_d, d.edge_attr_d = gd
        d.x_a, d.edge_index_a, d.edge_attr_a = ga
        d.qm = torch.tensor(qmz[k], dtype=torch.float).unsqueeze(0)
        d.y = torch.tensor([float(df["_delta"].iloc[k])], dtype=torch.float)
        d.g0 = torch.tensor([float(df["_baseline"].iloc[k])], dtype=torch.float)
        d.row_id = str(df["id"].iloc[k])
        out.append(d)
    print(f"  wldn graph build: {len(out)}/{len(df)} usable ({n_fail} dropped -- RDKit parse failure)",
          flush=True)
    return out


def make_loader(pairs, batch_size, shuffle):
    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, follow_batch=["x_p", "x_d", "x_a"])


def evl(model, ld, ym, ysd):
    model.eval()
    ps, ts, gs, ids_ = [], [], [], []
    for b in ld:
        b = b.to(dev)
        with torch.no_grad():
            o = model(b)
        ps.append(o.cpu())
        ts.append(b.ystd.cpu())
        gs.append(b.g0.cpu())
        ids_.extend(b.row_id)
    pdl = torch.cat(ps).numpy() * ysd + ym
    td = torch.cat(ts).numpy() * ysd + ym
    g0 = torch.cat(gs).numpy()
    yh, yt = g0 + pdl, g0 + td
    return float(np.abs(yt - yh).mean()), yh, yt, ids_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--feature-list", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out = args.outdir
    (out / "models").mkdir(parents=True, exist_ok=True)
    print("device", dev, flush=True)

    target_col = os.environ.get("CB_TARGET_COL", "dG_orca_kcal")
    baseline_col = os.environ.get("CB_BASELINE_COL", "dG_gxtb_kcal")
    print(f"target={target_col} baseline={baseline_col}", flush=True)

    df = pd.read_parquet(args.table)
    feats = json.loads(args.feature_list.read_text())
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"table missing {len(missing)} champion feature cols, e.g. {missing[:5]}")
    if "new_scaffold_split" not in df.columns:
        raise SystemExit("table has no new_scaffold_split column")

    df["_delta"] = df[target_col] - df[baseline_col]
    df["_baseline"] = df[baseline_col]

    if args.smoke:
        rng = np.random.default_rng(0)
        tr_idx = rng.choice(np.where((df["new_scaffold_split"] == "train").to_numpy())[0],
                             size=200, replace=False)
        va_idx = np.where((df["new_scaffold_split"] == "validation").to_numpy())[0][:50]
        te_idx = np.where((df["new_scaffold_split"] == "test").to_numpy())[0][:50]
        df = df.iloc[np.concatenate([tr_idx, va_idx, te_idx])].reset_index(drop=True)
        print(f"SMOKE MODE: subsampled to {len(df)} rows", flush=True)

    train_mask = (df["new_scaffold_split"] == "train").to_numpy()
    val_mask = (df["new_scaffold_split"] == "validation").to_numpy()
    test_mask = (df["new_scaffold_split"] == "test").to_numpy()
    print(f"rows: {len(df)} train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}", flush=True)

    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    med = Xdf[train_mask].median(numeric_only=True)
    Xz = Xdf.fillna(med).fillna(0.0)
    qm_mean = Xz[train_mask].mean().to_numpy()
    qm_std = Xz[train_mask].std().replace(0, 1).to_numpy()
    qm_std = np.where(qm_std == 0, 1.0, qm_std)
    qmz = ((Xz.to_numpy() - qm_mean) / qm_std).astype(np.float32)

    print("building product/donor/acceptor graphs...", flush=True)
    t1 = time.time()
    graphs = build_graphs(df, qmz)
    print(f"  built {len(graphs)}/{len(df)} in {time.time() - t1:.0f}s", flush=True)

    id_to_split = dict(zip(df["id"].astype(str), df["new_scaffold_split"]))
    tr_g = [g for g in graphs if id_to_split.get(g.row_id) == "train"]
    va_g = [g for g in graphs if id_to_split.get(g.row_id) == "validation"]
    te_g = [g for g in graphs if id_to_split.get(g.row_id) == "test"]
    print(f"  split: train={len(tr_g)} val={len(va_g)} test={len(te_g)}", flush=True)
    if len(tr_g) < 10 or len(te_g) < 5:
        raise SystemExit("too few usable rows -- aborting")

    ys = torch.tensor([g.y.item() for g in tr_g])
    ym, ysd = ys.mean().item(), ys.std().item()
    for g in graphs:
        g.ystd = (g.y - ym) / ysd

    ad, bd = graphs[0].x_p.shape[1], graphs[0].edge_attr_p.shape[1]
    nqm = len(feats)
    torch.manual_seed(args.seed)
    model = WLDNGNN(ad, bd, nqm, h=args.hidden, layers=args.layers, dropout=args.dropout).to(dev)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k "
          f"(node_dim={ad} edge_dim={bd} n_xd={nqm})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    tl = make_loader(tr_g, args.batch_size, True)
    vl = make_loader(va_g, args.batch_size, False)
    max_epochs = 2 if args.smoke else args.max_epochs
    best, best_state, pat = 1e9, None, 0
    t2 = time.time()
    for ep in range(max_epochs):
        model.train()
        for b in tl:
            b = b.to(dev)
            opt.zero_grad()
            loss = F.mse_loss(model(b), b.ystd)
            loss.backward()
            opt.step()
        vm = evl(model, vl, ym, ysd)[0]
        sch.step(vm)
        if vm < best - 1e-4:
            best, best_state, pat = vm, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if ep % 5 == 0 or pat == 0:
            print(f"  ep{ep:3d} val_MAE {vm:.3f} (best {best:.3f}) {time.time() - t2:.0f}s", flush=True)
        if pat >= args.patience:
            break
    model.load_state_dict(best_state)
    torch.save(best_state, out / "models" / "gnn_wldn_state.pt")

    te_mae, te_yhat, te_ytrue, te_ids = evl(model, make_loader(te_g, args.batch_size, False), ym, ysd)
    print(f"\n=== WLDN-GNN frozen holdout (n={len(te_g)}) MAE={te_mae:.4f} ===", flush=True)
    pd.DataFrame({"id": te_ids, "y_true": te_ytrue, "y_pred": te_yhat}).to_csv(
        out / "test_predictions.csv", index=False)

    result = {
        "built_by": "cross_benzoin/train_cross_gnn_wldn.py",
        "smoke": bool(args.smoke),
        "table": str(args.table), "target_col": target_col, "baseline_col": baseline_col,
        "n_train": len(tr_g), "n_val": len(va_g), "n_test": len(te_g),
        "n_dropped": len(df) - len(graphs),
        "node_dim": ad, "edge_dim": bd, "n_xd": nqm, "ym": ym, "ysd": ysd,
        "params": {"hidden": args.hidden, "layers": args.layers, "batch_size": args.batch_size,
                   "max_epochs": max_epochs, "lr": args.lr, "dropout": args.dropout,
                   "weight_decay": args.weight_decay, "seed": args.seed},
        "val_mae": best,
        "test_mae": te_mae,
        "cross_reference_champion_gnn": {"best_single_seed_mae": 0.556, "4seed_avg_mae": 0.531,
                                          "blend_mae": 0.528},
        "cross_reference_plain_crg": {"30seed_mean_mae": 0.572, "4seed_ensemble_mae": 0.543,
                                       "tuned_12seed_ensemble_mae": 0.535},
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"-> {out / 'result.json'}  ({result['runtime_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
