#!/usr/bin/env python
"""Condensed-reaction-graph (CRG) comparison point for cross-benzoin's dG GNN leg,
user-requested 2026-09-16 ("dG预测需要尝试使用反应图CRG等进行预测").

The deployed champion GNN (`train_cross_gnn.py`'s TripleGNN) encodes product,
donor, and acceptor as three SEPARATE, mutually-disconnected graphs, concatenating
their pooled embeddings only at the readout -- message passing can never let
donor-side and acceptor-side atoms actually influence each other's learned
embedding, even though in the real product they are covalently connected and
sterically/electronically interact (exactly why the champion schema needed
hand-built `interaction_*` features as a patch). This script instead builds ONE
connected graph -- the product's own molecular graph -- with two reaction-aware
feature additions (see crg_builder.py for the construction + validation):
  - per-atom: [donor_side, acceptor_side, is_core] one-hot (which side of the new
    bond an atom sits on, via BFS-split of the product graph itself)
  - per-bond: is_new_bond flag on the ketC-carbC edge (the bond that does not
    exist in either separate reactant)
Single Enc() (same GINEConv block as TripleGNN's per-branch encoder) over this
one graph, pooled, concatenated with the same 257-feat champion x_d schema.

crg_builder.build_crg() drops ~2.5% of rows (ambiguous/absent reactive-core
SMARTS match); this script trains + evaluates on whatever subset resolves, same
graceful-drop convention used elsewhere in this project.

Same target, x_d schema, and `new_scaffold_split`-based split as
train_cross_gnn_chemprop.py (see that file's docstring for why this project
avoids train_cross_delta.pair_split_labels() post-candidates_v3-retirement).

Env: /home/schen3/venv/nequip (torch 2.11 + PyTorch Geometric 2.7, the SAME env
the deployed champion TripleGNN is trained in -- so a chemistry/architecture-only
comparison, no env confound).

Usage
  # CPU correctness smoke test:
  /home/schen3/venv/nequip/bin/python cross_benzoin/train_cross_gnn_crg.py \
      --table data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet \
      --feature-list data/cross_benzoin/feature_list_257_no_nCHO_v2.json \
      --outdir /tmp/crg_smoke --smoke

  # real run (GPU, via sbatch submit_cross_gnn_crg.sh):
  CB_TARGET_COL=dG_r2scan_kcal CB_BASELINE_COL=dG_b973c_kcal \
  /home/schen3/venv/nequip/bin/python cross_benzoin/train_cross_gnn_crg.py \
      --table ... --feature-list ... --outdir data/cross_benzoin/cross_round10/gnn_crg_10rounds_b973c_v1
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crg_builder import build_crg  # noqa: E402

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


class CRGGNN(nn.Module):
    """Single encoder over the condensed product graph + QM-scalar channel."""

    def __init__(self, ad, bd, nqm, h=128, layers=4, dropout=0.1):
        super().__init__()
        self.enc = Enc(ad, bd, h, layers)
        din = 2 * h + nqm
        self.head = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, b):
        hC = self.enc(b.x, b.edge_index, b.edge_attr, b.batch)
        h = torch.cat([hC, b.qm.view(-1, b.qm.shape[-1])], 1)
        return self.head(h).squeeze(-1)


def build_graphs(df: pd.DataFrame, qmz: np.ndarray):
    out = []
    n_fail = 0
    for k in range(len(df)):
        r = build_crg(df["smiles"].iloc[k])
        if r is None:
            n_fail += 1
            continue
        x, edge_index, edge_attr, _core = r
        d = Data(x=torch.tensor(x, dtype=torch.float),
                  edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                  edge_attr=torch.tensor(edge_attr, dtype=torch.float))
        d.qm = torch.tensor(qmz[k], dtype=torch.float).unsqueeze(0)
        d.y = torch.tensor([float(df["_delta"].iloc[k])], dtype=torch.float)
        d.g0 = torch.tensor([float(df["_baseline"].iloc[k])], dtype=torch.float)
        d.row_id = str(df["id"].iloc[k])
        out.append(d)
    print(f"  crg build: {len(out)}/{len(df)} usable ({n_fail} dropped -- ambiguous/no reactive-core match)",
          flush=True)
    return out


def make_loader(pairs, batch_size, shuffle):
    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle)


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

    train_mask = df["new_scaffold_split"].isin(["train", "mixed"]).to_numpy()  # fold mixed into training, matching champion TripleGNN's convention (2026-09-17 fix)
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

    print("building condensed reaction graphs (product only)...", flush=True)
    t1 = time.time()
    graphs = build_graphs(df, qmz)
    print(f"  built {len(graphs)}/{len(df)} in {time.time() - t1:.0f}s", flush=True)

    id_to_split = dict(zip(df["id"].astype(str), df["new_scaffold_split"]))
    tr_g = [g for g in graphs if id_to_split.get(g.row_id) in ("train", "mixed")]
    va_g = [g for g in graphs if id_to_split.get(g.row_id) == "validation"]
    te_g = [g for g in graphs if id_to_split.get(g.row_id) == "test"]
    print(f"  split: train={len(tr_g)} val={len(va_g)} test={len(te_g)}", flush=True)
    if len(tr_g) < 10 or len(te_g) < 5:
        raise SystemExit("too few usable rows after CRG drop -- aborting")

    ys = torch.tensor([g.y.item() for g in tr_g])
    ym, ysd = ys.mean().item(), ys.std().item()
    for g in graphs:
        g.ystd = (g.y - ym) / ysd

    ad, bd = graphs[0].x.shape[1], graphs[0].edge_attr.shape[1]
    nqm = len(feats)
    torch.manual_seed(args.seed)
    model = CRGGNN(ad, bd, nqm, h=args.hidden, layers=args.layers, dropout=args.dropout).to(dev)
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
    torch.save(best_state, out / "models" / "gnn_crg_state.pt")

    te_mae, te_yhat, te_ytrue, te_ids = evl(model, make_loader(te_g, args.batch_size, False), ym, ysd)
    print(f"\n=== CRG-GNN frozen holdout (n={len(te_g)}) MAE={te_mae:.4f} ===", flush=True)
    pd.DataFrame({"id": te_ids, "y_true": te_ytrue, "y_pred": te_yhat}).to_csv(
        out / "test_predictions.csv", index=False)

    result = {
        "built_by": "cross_benzoin/train_cross_gnn_crg.py",
        "smoke": bool(args.smoke),
        "table": str(args.table), "target_col": target_col, "baseline_col": baseline_col,
        "n_train": len(tr_g), "n_val": len(va_g), "n_test": len(te_g),
        "n_dropped_no_core_match": len(df) - len(graphs),
        "node_dim": ad, "edge_dim": bd, "n_xd": nqm, "ym": ym, "ysd": ysd,
        "params": {"hidden": args.hidden, "layers": args.layers, "batch_size": args.batch_size,
                   "max_epochs": max_epochs, "lr": args.lr, "dropout": args.dropout,
                   "weight_decay": args.weight_decay, "seed": args.seed},
        "val_mae": best,
        "test_mae": te_mae,
        "cross_reference_champion_gnn": {"best_single_seed_mae": 0.556, "4seed_avg_mae": 0.531,
                                          "blend_mae": 0.528},
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"-> {out / 'result.json'}  ({result['runtime_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
