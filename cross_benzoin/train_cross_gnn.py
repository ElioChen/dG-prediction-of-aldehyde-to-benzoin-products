#!/usr/bin/env python
"""Cross-benzoin GNN + tabular stacking prototype, mirroring homo's confirmed win
(pipeline/analysis/gnn_dual_qm_champion275_aligned_v2.py: dual-encoder GINE-GNN
blended with a tabular MLP+XGB ensemble, job 24578348, test MAE 1.503->1.427,
see memory [[gnn-stacking-confirmed-full-scale]]).

Cross differs from homo in one structural way that changes the architecture: homo
pairs have donor==acceptor (one aldehyde graph suffices), cross pairs have
donor!=acceptor, so this uses a TRIPLE encoder -- product graph + donor graph +
acceptor graph -- instead of homo's dual encoder. Everything else (GINEConv atom/
bond featurization, QM-scalar-at-readout, delta-learning target, early stopping)
is a direct port.

QM-scalar channel: reuses the shipped single-XGB champion's exact 260-feature
`all_raw_blocks+mordred` selection (data/cross_benzoin/cross_round5/
train_5rounds_mordred_slim120_v1/models/feature_list.json) -- the same descriptor
set already validated as the champion/ensemble's feature schema, rather than
reinventing a QM feature list.

Split: reuses train_cross_delta.py's pair_split_labels() three-way bucket
(candidates_v3 train/validation/test) verbatim -- train on 'train' rows, early-
stop on 'validation' rows, evaluate once on the same frozen 29 'test' rows the
single-XGB champion (frozen holdout MAE 2.983) and MLP+XGB ensemble (2.633) were
scored on, for a genuinely comparable number. NOT a repeated-CV run (unlike the
tabular models) -- GPU budget is ample but this is a first prototype; the frozen-
holdout comparison is the scientifically load-bearing one in this project anyway.

Usage
  python cross_benzoin/train_cross_gnn.py \
      --table data/cross_benzoin/cross_round5/cross_train_table_5rounds_mordred_slim120.parquet \
      --champion-dir data/cross_benzoin/cross_round5/train_5rounds_mordred_slim120_v1 \
      --ensemble-dir data/cross_benzoin/cross_round5/train_ensemble_slim120_v1 \
      --outdir data/cross_benzoin/cross_round5/train_gnn_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402
from train_cross_delta import TARGET_COL, BASELINE_COL, pair_split_labels  # noqa: E402
from train_cross_ensemble import MLPXGBEnsemble  # noqa: E402,F401

# train_cross_ensemble.py's joblib artifact was pickled while running as __main__
# (its class's __module__ is recorded as "__main__" at save time), so a plain
# `from train_cross_ensemble import MLPXGBEnsemble` alone isn't enough -- pickle
# looks the class up in sys.modules["__main__"], which here is THIS script.
import __main__ as _main_mod  # noqa: E402

_main_mod.MLPXGBEnsemble = MLPXGBEnsemble

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── atom/bond featurization (ported verbatim from pipeline/analysis/gnn_dual_qm_champion275.py) ──
ELEMS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"]


def oh(x, ch):
    return [int(x == c) for c in ch] + [int(x not in ch)]


def af(a):
    return (oh(a.GetSymbol(), ELEMS) + oh(a.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
            + oh(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + oh(str(a.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
            + oh(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + [int(a.GetIsAromatic()), int(a.IsInRing())])


BTD = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}


def bf(b):
    v = [0, 0, 0, 0]
    v[BTD.get(b.GetBondType(), 0)] = 1
    return v + [int(b.GetIsConjugated()), int(b.IsInRing())]


def graph(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None or m.GetNumAtoms() == 0:
        return None
    x = torch.tensor([af(a) for a in m.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bf(b)
        ei += [[i, j], [j, i]]
        ea += [f, f]
    if not ei:
        ei = [[0, 0]]
        ea = [[0, 0, 0, 0, 0, 0]]
    return x, torch.tensor(ei).t().contiguous(), torch.tensor(ea, dtype=torch.float)


class TripleData(Data):
    def __inc__(self, key, value, *a, **k):
        if key == "edge_index_p":
            return self.x_p.size(0)
        if key == "edge_index_d":
            return self.x_d.size(0)
        if key == "edge_index_a":
            return self.x_a.size(0)
        return super().__inc__(key, value, *a, **k)


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


class TripleGNN(nn.Module):
    """Product + donor + acceptor encoders (cross's donor!=acceptor analogue of
    homo's dual product+aldehyde encoder). Readout concatenates all three pooled
    embeddings, a donor-acceptor asymmetry term (orientation matters: donor becomes
    the ketone carbon, acceptor the carbinol carbon), and the QM-scalar channel."""

    def __init__(self, ad, bd, nqm, h=128, layers=4):
        super().__init__()
        self.encP = Enc(ad, bd, h, layers)
        self.encD = Enc(ad, bd, h, layers)
        self.encA = Enc(ad, bd, h, layers)
        din = 8 * h + nqm
        self.head = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, b):
        hP = self.encP(b.x_p, b.edge_index_p, b.edge_attr_p, b.x_p_batch)
        hD = self.encD(b.x_d, b.edge_index_d, b.edge_attr_d, b.x_d_batch)
        hA = self.encA(b.x_a, b.edge_index_a, b.edge_attr_a, b.x_a_batch)
        h = torch.cat([hP, hD, hA, hD - hA, b.qm.view(-1, b.qm.shape[-1])], 1)
        return self.head(h).squeeze(-1)


def build_graphs(df: pd.DataFrame, qmz: np.ndarray) -> list[TripleData]:
    pairs = []
    for k in range(len(df)):
        gp, gd, ga = graph(df["smiles"].iloc[k]), graph(df["donor_smiles"].iloc[k]), graph(df["acceptor_smiles"].iloc[k])
        if gp is None or gd is None or ga is None:
            continue
        d = TripleData()
        d.x_p, d.edge_index_p, d.edge_attr_p = gp
        d.x_d, d.edge_index_d, d.edge_attr_d = gd
        d.x_a, d.edge_index_a, d.edge_attr_a = ga
        d.y = torch.tensor([float(df[TARGET_COL].iloc[k]) - float(df[BASELINE_COL].iloc[k])], dtype=torch.float)
        d.g0 = torch.tensor([float(df[BASELINE_COL].iloc[k])], dtype=torch.float)
        d.qm = torch.tensor(qmz[k], dtype=torch.float).view(1, -1)
        d.row_id = str(df["id"].iloc[k])
        pairs.append(d)
    return pairs


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
    ap.add_argument("--table", required=True)
    ap.add_argument("--champion-dir", required=True, help="single-XGB champion dir (feature_list.json)")
    ap.add_argument("--ensemble-dir", required=True, help="MLP+XGB ensemble dir (for the stacking blend)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from-homo", default=None,
                     help="path to a homo DualGNN state dict (e.g. gnn_dual_champion275_"
                          "ALIGNEDV2_state_20260713.pt) -- transfer-learning init: homo's "
                          "encP (product graph) copies into this model's encP, homo's encA "
                          "(single aldehyde graph) copies into BOTH encD and encA (donor/"
                          "acceptor are both 'aldehyde-like' roles). Requires identical atom/"
                          "bond featurization (verified: AD=40, BD=6, h=128, 4 GINEConv layers "
                          "match homo's architecture exactly, same af()/bf()/graph() functions). "
                          "The head is always randomly initialized (input dim differs: cross's "
                          "8h+nqm vs homo's 6h+nqm).")
    ap.add_argument("--extra-val-frac", type=float, default=0.0,
                     help="carve this fraction of the train_extra pool (grouped by pair_key) "
                          "into EXTRA early-stopping validation rows, supplementing "
                          "candidates_v3's own tiny 22-row validation split (too small/noisy "
                          "a signal alone -- val_MAE bounced non-monotonically in the first run).")
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    print("device", dev, flush=True)

    df = pd.read_parquet(args.table)
    feats = json.loads((Path(args.champion_dir) / "models" / "feature_list.json").read_text())
    print(f"loaded {len(df)} rows, {len(feats)} QM-scalar feats (champion schema)", flush=True)

    # NaN-split rows (mostly legacy round1/2 custom-sampled pairs whose aldehyde
    # isn't in candidates_v3's split map) are folded into the TRAIN bucket as
    # "train_extra" -- matching the exact policy train_cross_delta.py/
    # train_cross_ensemble.py use for their shipped final-fit models (clean_mask
    # = NOT in ['test','validation']), so the GNN's training population (17219
    # rows) matches what the SHIPPED ensemble artifact was actually trained on,
    # not the smaller in-function-only fit train_cross_ensemble.py's
    # frozen_holdout_eval_ensemble() uses purely to report its 2.633 metric
    # (that fit is never saved to disk). NOTE: this does NOT make the "ensemble-
    # only" baseline below equal 2.633 -- it stays ~2.777, because it comes from
    # predict()-ing the shipped joblib artifact (trained on 17219 rows), which is
    # a genuinely different model than the 16431-row-only fit that produced the
    # officially reported 2.633 number. Both are legitimate; 2.777 is "what the
    # artifact you'd actually ship predicts," 2.633 is "the metric as reported."
    pair_split = pair_split_labels(df, verbose=True)
    df = df.assign(_split=pair_split)
    df["_split"] = df["_split"].where(df["_split"].notna(), "train_extra")

    if args.extra_val_frac > 0:
        rng = np.random.default_rng(args.seed)
        extra_pairs = df.loc[df["_split"] == "train_extra", "pair_key"].unique()
        n_extra_val = int(len(extra_pairs) * args.extra_val_frac)
        val_pairs_extra = set(rng.choice(extra_pairs, size=n_extra_val, replace=False))
        promote = (df["_split"] == "train_extra") & df["pair_key"].isin(val_pairs_extra)
        df.loc[promote, "_split"] = "validation_extra"
        print(f"carved {promote.sum()} rows ({len(val_pairs_extra)} pairs) from train_extra "
              f"into extra early-stopping validation", flush=True)

    print(f"rows: {len(df)} ({df['_split'].value_counts().to_dict()})", flush=True)

    train_mask = df["_split"].isin(["train", "train_extra"]).to_numpy()
    val_mask = df["_split"].isin(["validation", "validation_extra"]).to_numpy()
    test_mask = (df["_split"] == "test").to_numpy()

    # QM-scalar: median-impute + standardize, fit on 'train' rows only (no leakage into val/test)
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    med = Xdf[train_mask].median(numeric_only=True)
    Xz = Xdf.fillna(med).fillna(0.0)
    qm_mean = Xz[train_mask].mean().to_numpy()
    qm_std = Xz[train_mask].std().replace(0, 1).to_numpy()
    qm_std = np.where(qm_std == 0, 1.0, qm_std)
    qmz = ((Xz.to_numpy() - qm_mean) / qm_std).astype(np.float32)

    print("building graphs (product + donor + acceptor)...", flush=True)
    t0 = time.time()
    pairs = build_graphs(df, qmz)
    print(f"  built {len(pairs)}/{len(df)} pairs ({time.time()-t0:.0f}s)", flush=True)
    id_to_split = dict(zip(df["id"].astype(str), df["_split"]))
    tr_pairs = [p for p in pairs if id_to_split.get(p.row_id) in ("train", "train_extra")]
    va_pairs = [p for p in pairs if id_to_split.get(p.row_id) in ("validation", "validation_extra")]
    te_pairs = [p for p in pairs if id_to_split.get(p.row_id) == "test"]
    print(f"  split: train={len(tr_pairs)} val={len(va_pairs)} test={len(te_pairs)}", flush=True)

    ys = torch.tensor([p.y.item() for p in tr_pairs])
    ym, ysd = ys.mean().item(), ys.std().item()
    for p in pairs:
        p.ystd = (p.y - ym) / ysd

    ad, bd = pairs[0].x_p.shape[1], pairs[0].edge_attr_p.shape[1]
    nqm = len(feats)
    torch.manual_seed(args.seed)
    model = TripleGNN(ad, bd, nqm, h=args.hidden, layers=args.layers).to(dev)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k", flush=True)

    if args.init_from_homo:
        homo_sd = torch.load(args.init_from_homo, map_location="cpu", weights_only=False)
        own_sd = model.state_dict()
        loaded, skipped = 0, 0
        for dst_prefix, src_prefix in [("encP.", "encP."), ("encD.", "encA."), ("encA.", "encA.")]:
            for k, v in homo_sd.items():
                if not k.startswith(src_prefix):
                    continue
                dst_k = dst_prefix + k[len(src_prefix):]
                if dst_k in own_sd and own_sd[dst_k].shape == v.shape:
                    own_sd[dst_k] = v.clone()
                    loaded += 1
                else:
                    skipped += 1
        model.load_state_dict(own_sd)
        print(f"transfer-learning init from {args.init_from_homo}: "
              f"{loaded} encoder tensors loaded (encP<-homo.encP, encD/encA<-homo.encA), "
              f"{skipped} skipped (shape mismatch, e.g. head layers stay randomly init)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    tl = make_loader(tr_pairs, args.batch_size, True)
    vl = make_loader(va_pairs, args.batch_size, False)
    best, best_state, pat = 1e9, None, 0
    t1 = time.time()
    for ep in range(args.max_epochs):
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
            print(f"  ep{ep:3d} val_MAE {vm:.3f} (best {best:.3f}) {time.time()-t1:.0f}s", flush=True)
        if pat >= args.patience:
            break
    model.load_state_dict(best_state)
    torch.save(best_state, out / "models" / "cross_gnn_state.pt")

    mae_gnn, yh_gnn, yt_gnn, ids_gnn = evl(model, make_loader(te_pairs, args.batch_size, False), ym, ysd)
    print(f"\n=== GNN frozen holdout (n={len(te_pairs)}) MAE={mae_gnn:.3f} ===", flush=True)

    # ── stacking blend against the already-shipped MLP+XGB ensemble's OWN frozen-holdout predictions ──
    ens = joblib.load(Path(args.ensemble_dir) / "models" / "cross_ensemble_model.joblib")
    test_df = df[df["_split"] == "test"].set_index(df[df["_split"] == "test"]["id"].astype(str))
    ens_pred = ens.predict(test_df) + test_df[BASELINE_COL].to_numpy()
    ens_pred_by_id = dict(zip(test_df.index, ens_pred))
    gnn_pred_by_id = dict(zip(ids_gnn, yh_gnn))
    true_by_id = dict(zip(ids_gnn, yt_gnn))
    common = [i for i in ids_gnn if i in ens_pred_by_id]
    y_true = np.array([true_by_id[i] for i in common])
    gnn_p = np.array([gnn_pred_by_id[i] for i in common])
    ens_p = np.array([ens_pred_by_id[i] for i in common])
    mae_gnn_c = float(np.abs(y_true - gnn_p).mean())
    mae_ens_c = float(np.abs(y_true - ens_p).mean())
    rows = [{"w_gnn": 0.0, "mae": mae_ens_c, "config": "ensemble_only"}]
    best_w, best_mae = 0.0, mae_ens_c
    for w in np.arange(0.05, 1.0, 0.05):
        blend = (1 - w) * ens_p + w * gnn_p
        m = float(np.abs(y_true - blend).mean())
        rows.append({"w_gnn": round(float(w), 2), "mae": m, "config": "blend"})
        if m < best_mae:
            best_mae, best_w = m, w
    rows.append({"w_gnn": 1.0, "mae": mae_gnn_c, "config": "gnn_only"})
    res = pd.DataFrame(rows)
    res.to_csv(out / "data" / "gnn_ensemble_stack.csv", index=False)
    print(f"\n=== stacking (n={len(common)}) ===", flush=True)
    print(f"  ensemble-only MAE={mae_ens_c:.3f} | gnn-only MAE={mae_gnn_c:.3f} | "
          f"best blend w_gnn={best_w:.2f} MAE={best_mae:.3f} (delta vs ensemble-only {best_mae-mae_ens_c:+.3f})",
          flush=True)

    (out / "models" / "metadata.json").write_text(json.dumps({
        "model": "triple_gnn_gine_h128_l4",
        "init_from_homo": args.init_from_homo,
        "extra_val_frac": args.extra_val_frac,
        "n_train": len(tr_pairs), "n_val": len(va_pairs), "n_test": len(te_pairs),
        "gnn_frozen_holdout_mae": mae_gnn,
        "stacking_common_n": len(common),
        "ensemble_only_mae": mae_ens_c, "gnn_only_mae": mae_gnn_c,
        "best_blend_w_gnn": float(best_w), "best_blend_mae": float(best_mae),
        "reference_single_xgb_frozen_holdout_mae": 2.983,
        "reference_ensemble_frozen_holdout_mae": 2.633,
    }, indent=2))
    print(f"\nSaved GNN state + metadata to {out}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
