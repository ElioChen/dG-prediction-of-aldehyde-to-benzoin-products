#!/usr/bin/env python
"""GINE GNN Δ-learning: predict (DFT − g-xTB) from the product SMILES graph.

Final ΔG_pred = dG_gxtb_kcal + GNN(structure). Head-to-head with the GBT Δ-model
(test MAE 2.46) on the SAME random 70/20/10 split (seed 42). Earlier finding
[gnn-delta-result] (D-MPNN lost to trees) was at n=474; here n≈136k, a fair test.
Then PREDICT the correction for the whole library and write corrected ΔG.

Run on a GPU node (envs/gnn: torch + PyG2.8 + rdkit). Writes metrics + parity (new files).
"""
import glob
import os
import time

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

ROOT = "/scratch-shared/schen3/benzoin-dg"
PROD = f"{ROOT}/data/cross_benzoin/homo_v6/products_all.csv"
DFTDIR = f"{ROOT}/data/raw/dft_sp_funnelv3"
OUT = f"{ROOT}/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
SUF = os.environ.get("SUFFIX", "")          # e.g. "_migquick" to avoid clobbering the H100 sweep
MAXEP = int(os.environ.get("MAXEP", "150"))
CACHE = f"{OUT}/gnn_delta_cache{SUF}.pt"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, flush=True)

# ---------- featurizer (same scheme as screen_v6 gnn_dG.py) ----------
ELEMS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"]
def onehot(x, ch): return [int(x == c) for c in ch] + [int(x not in ch)]
def atom_feat(a):
    return (onehot(a.GetSymbol(), ELEMS) + onehot(a.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
            + onehot(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + onehot(str(a.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
            + onehot(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + [int(a.GetIsAromatic()), int(a.IsInRing())])
BT = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
def bond_feat(b):
    bt = [0, 0, 0, 0]; bt[BT.get(b.GetBondType(), 0)] = 1
    return bt + [int(b.GetIsConjugated()), int(b.IsInRing())]

def mol_to_graph(smi, y, g0):
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0: return None
    x = torch.tensor([atom_feat(a) for a in m.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); f = bond_feat(b)
        ei += [[i, j], [j, i]]; ea += [f, f]
    if not ei:
        ei = [[0, 0]]; ea = [[0, 0, 0, 0, 0, 0]]
    return Data(x=x, edge_index=torch.tensor(ei).t().contiguous(),
                edge_attr=torch.tensor(ea, dtype=torch.float),
                y=torch.tensor([y], dtype=torch.float),
                g0=torch.tensor([g0], dtype=torch.float))   # g-xTB baseline carried along

# ---------- load + build (cached) ----------
if os.path.exists(CACHE):
    blob = torch.load(CACHE, weights_only=False)
    graphs = blob["graphs"]
    print("loaded cache", len(graphs), flush=True)
else:
    fs = sorted(glob.glob(f"{DFTDIR}/chunk_*.csv"))
    dft = pd.concat([pd.read_csv(f, usecols=["id", "dG_orca_kcal"]) for f in fs], ignore_index=True)
    dft = dft.dropna(subset=["dG_orca_kcal"]).drop_duplicates("id")
    p = pd.read_csv(PROD, usecols=["id", "smiles", "dG_gxtb_kcal"], low_memory=False)
    df = p.merge(dft, on="id", how="inner").dropna(subset=["smiles", "dG_gxtb_kcal", "dG_orca_kcal"])
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    print("building", len(df), "graphs", flush=True)
    graphs = []
    t0 = time.time()
    for k, (smi, g0, yv) in enumerate(zip(df["smiles"], df["dG_gxtb_kcal"], df["dG_orca_kcal"])):
        g = mol_to_graph(smi, float(yv) - float(g0), float(g0))   # target = Δ = DFT − g-xTB
        if g is not None:
            graphs.append(g)
        if k % 20000 == 0:
            print(f"  {k} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"graphs": graphs}, CACHE)
    print("built+cached", len(graphs), flush=True)

N = len(graphs)
# ---------- random 70/20/10 split (seed 42, matches GBT comparison) ----------
rng = np.random.default_rng(42)
idx = rng.permutation(N)
ntr, nva = int(.7 * N), int(.2 * N)
tr, va, te = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
print(f"split train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)

ys = torch.tensor([graphs[i].y.item() for i in tr])
ym, ysd = ys.mean().item(), ys.std().item()
for g in graphs:
    g.ystd = (g.y - ym) / ysd
def loader(ix, sh): return DataLoader([graphs[i] for i in ix], batch_size=256, shuffle=sh)
tl, vl, sl = loader(tr, True), loader(va, False), loader(te, False)

# ---------- GINE ----------
class GNN(nn.Module):
    def __init__(self, ad, bd, h=128, L=4):
        super().__init__()
        self.ne = nn.Linear(ad, h); self.ee = nn.Linear(bd, h)
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        for _ in range(L):
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.convs.append(GINEConv(mlp, edge_dim=h)); self.bns.append(nn.BatchNorm1d(h))
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def forward(self, d):
        x = self.ne(d.x); e = self.ee(d.edge_attr)
        for c, bn in zip(self.convs, self.bns):
            x = x + F.relu(bn(c(x, d.edge_index, e)))
        g = torch.cat([global_mean_pool(x, d.batch), global_add_pool(x, d.batch)], 1)
        return self.head(g).squeeze(-1)

ad = graphs[0].x.shape[1]; bd = graphs[0].edge_attr.shape[1]
model = GNN(ad, bd).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
print(f"atom_dim {ad} bond_dim {bd} params {sum(p.numel() for p in model.parameters())}", flush=True)

def run(ld, train):
    model.train(train); ps, ts, gs = [], [], []
    tot = 0.0
    for b in ld:
        b = b.to(dev)
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(b); loss = F.mse_loss(out, b.ystd)
            if train: loss.backward(); opt.step()
        tot += loss.item() * b.num_graphs
        ps.append(out.detach().cpu()); ts.append(b.ystd.detach().cpu()); gs.append(b.g0.detach().cpu())
    pred_delta = torch.cat(ps).numpy() * ysd + ym       # predicted Δ
    true_delta = torch.cat(ts).numpy() * ysd + ym
    g0 = torch.cat(gs).numpy()
    # absolute ΔG = g-xTB + Δ ; metrics on absolute DFT scale
    yhat = g0 + pred_delta; ytrue = g0 + true_delta
    r2 = 1 - ((ytrue - yhat) ** 2).sum() / ((ytrue - ytrue.mean()) ** 2).sum()
    return (tot / len(ld.dataset), r2, float(np.sqrt(((ytrue - yhat) ** 2).mean())),
            float(np.abs(ytrue - yhat).mean()), ytrue, yhat)

best_v, best_state, patience = 1e9, None, 0
for ep in range(MAXEP):
    run(tl, True)
    vloss, vr2, _, vmae, _, _ = run(vl, False)
    sched.step(vmae)
    if vmae < best_v:
        best_v, best_state, patience = vmae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    else:
        patience += 1
    if ep % 5 == 0 or patience == 0:
        print(f"ep{ep:3d} val_MAE {vmae:.3f} R2 {vr2:.3f} (best {best_v:.3f})", flush=True)
    if patience >= 18:
        print("early stop", flush=True); break

model.load_state_dict(best_state)
_, r2, rmse, mae, t, p = run(sl, False)
print(f"\nTEST  random70/20/10  R2={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}  (vs GBT 2.46)", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 5.6))
ax.hexbin(t, p, gridsize=70, cmap="viridis", bins="log", mincnt=1, extent=(-25, 30, -25, 30))
ax.plot([-25, 30], [-25, 30], "w--", lw=1); ax.set_xlim(-25, 30); ax.set_ylim(-25, 30)
ax.set_xlabel("DFT r2SCAN-3c ΔG (kcal/mol)"); ax.set_ylabel("g-xTB + GNN(Δ) (kcal/mol)")
ax.set_title(f"GINE Δ-GNN vs DFT (TEST)\nR²={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}  (GBT=2.46)")
fig.tight_layout(); fig.savefig(f"{OUT}/25_parity_gnn_delta{SUF}.png", dpi=150)
print(f"wrote 25_parity_gnn_delta{SUF}.png", flush=True)

with open(f"{OUT}/gnn_delta_metrics{SUF}.txt", "w") as fh:
    fh.write(f"GINE Δ-GNN (DFT−g-xTB), random 70/20/10 seed42, n={N}\n")
    fh.write(f"TEST  R2={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}  | GBT Δ MAE=2.46 ridge=2.87 raw g-xTB=4.35\n")

# ---------- PREDICT correction for the WHOLE library ----------
print("predicting full-library correction ...", flush=True)
pf = pd.read_csv(PROD, usecols=["id", "smiles", "dG_gxtb_kcal"], low_memory=False)
pf = pf.dropna(subset=["smiles", "dG_gxtb_kcal"]).reset_index(drop=True)
model.eval()
preds = np.full(len(pf), np.nan)
buf, pos = [], []
BATCH = 512
def flush_buf():
    if not buf: return
    dl = DataLoader(buf, batch_size=BATCH, shuffle=False)
    out = []
    for b in dl:
        b = b.to(dev)
        with torch.no_grad():
            out.append(model(b).cpu().numpy() * ysd + ym)
    out = np.concatenate(out)
    for q, v in zip(pos, out): preds[q] = v
    buf.clear(); pos.clear()
for k, (smi, g0) in enumerate(zip(pf["smiles"], pf["dG_gxtb_kcal"])):
    g = mol_to_graph(smi, 0.0, float(g0))
    if g is not None:
        buf.append(g); pos.append(k)
    if len(buf) >= 20000: flush_buf()
flush_buf()
pf["delta_gnn_pred"] = preds
pf["dG_gxtb_gnn_corrected"] = pf["dG_gxtb_kcal"] + preds
pf[["id", "dG_gxtb_kcal", "delta_gnn_pred", "dG_gxtb_gnn_corrected"]].to_csv(
    f"{OUT}/products_gxtb_gnn_corrected{SUF}.csv", index=False)
print(f"wrote products_gxtb_gnn_corrected{SUF}.csv ({pf['dG_gxtb_gnn_corrected'].notna().sum():,} corrected)", flush=True)
print("DONE", flush=True)
