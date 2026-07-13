#!/usr/bin/env python
"""Multi-architecture GNN study: predict Δ=(DFT − g-xTB) from product SMILES.

Trains several GNN architectures sequentially on ONE GPU, same random 70/20/10 (seed 42),
same 136k DFT-labelled products, same Δ-learning target -> fair head-to-head vs the
GBT Δ-model (test MAE 2.46). Architectures:
  gine        : edge-aware GIN (baseline)
  gine_big    : GINE h256 L5 (capacity)
  gat         : graph attention (edge-aware)
  gcn         : graph conv (ignores bond features) — ablation
  nnconv      : edge-conditioned MPNN (closest to D-MPNN)
  gine_hybrid : GINE + 34 QM descriptors concatenated at readout (does QM help the graph?)
Picks the best by val MAE, writes a comparison table, best-model parity, and the
full-library corrected ΔG from the best model.

GPU node, envs/gnn (torch + PyG2.8 + rdkit).
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
from torch_geometric.nn import (GATv2Conv, GCNConv, GINEConv, NNConv,
                                global_add_pool, global_mean_pool)

RDLogger.DisableLog("rdApp.*")
ROOT = "/scratch-shared/schen3/benzoin-dg"
PROD = f"{ROOT}/data/cross_benzoin/homo_v6/products_all.csv"
DFTDIR = f"{ROOT}/data/raw/dft_sp_funnelv3"
CLSF = f"{ROOT}/data/cross_benzoin/homo_v6/aldehyde_class.parquet"
OUT = f"{ROOT}/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
SCOPE = os.environ.get("SCOPE", "all")          # all | aromatic | aliphatic (by donor aldehyde class)
SUF = os.environ.get("SUFFIX", "")              # output/cache suffix to avoid clobber
ARCHS = os.environ.get("ARCHS", "")             # comma list to restrict configs, e.g. "gine_hybrid"
CACHE = f"{OUT}/gnn_arch_cache{SUF}_full.pt"  # full 100% labels
os.makedirs(OUT, exist_ok=True)
print(f"SCOPE={SCOPE} SUFFIX='{SUF}' ARCHS='{ARCHS}'", flush=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, flush=True)

QM = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta", "xtb_omega",
      "xtb_dipole", "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO",
      "mulliken_hydH", "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb", "fukui_plus_ketC",
      "fukui_minus_ketC", "dual_ketC", "fukui_plus_carbC", "fukui_minus_carbC", "dual_carbC",
      "vbur_ketC", "vbur_carbC", "sterimol_L", "sterimol_B1", "sterimol_B5", "SASA_total",
      "P_int", "pa_ketO", "hb_dist", "hb_angle", "dih_core"]

ELEMS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"]
def onehot(x, ch): return [int(x == c) for c in ch] + [int(x not in ch)]
def atom_feat(a):
    return (onehot(a.GetSymbol(), ELEMS) + onehot(a.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
            + onehot(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + onehot(str(a.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
            + onehot(a.GetTotalNumHs(), [0, 1, 2, 3, 4]) + [int(a.GetIsAromatic()), int(a.IsInRing())])
BT = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
def bond_feat(b):
    bt = [0, 0, 0, 0]; bt[BT.get(b.GetBondType(), 0)] = 1
    return bt + [int(b.GetIsConjugated()), int(b.IsInRing())]
def mol_to_graph(smi, delta, g0, qm):
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0: return None
    x = torch.tensor([atom_feat(a) for a in m.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); f = bond_feat(b)
        ei += [[i, j], [j, i]]; ea += [f, f]
    if not ei: ei = [[0, 0]]; ea = [[0, 0, 0, 0, 0, 0]]
    return Data(x=x, edge_index=torch.tensor(ei).t().contiguous(),
                edge_attr=torch.tensor(ea, dtype=torch.float),
                y=torch.tensor([delta], dtype=torch.float),
                g0=torch.tensor([g0], dtype=torch.float),
                qm=torch.tensor(qm, dtype=torch.float).view(1, -1))

# ---------- build / cache ----------
if os.path.exists(CACHE):
    blob = torch.load(CACHE, weights_only=False)
    graphs, qm_mean, qm_std = blob["graphs"], blob["qm_mean"], blob["qm_std"]
    print("loaded cache", len(graphs), flush=True)
else:
    dft = pd.read_parquet(f"{DFTDIR}/dft_labels_all.parquet", columns=["id","dG_orca_kcal"]).dropna().drop_duplicates("id")  # consolidated 100%
    p = pd.read_csv(PROD, usecols=["id", "smiles", "dG_gxtb_kcal"] + QM, low_memory=False)
    df = p.merge(dft, on="id", how="inner").dropna(subset=["smiles", "dG_gxtb_kcal", "dG_orca_kcal"] + QM)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    if SCOPE != "all":
        cl = pd.read_parquet(CLSF)
        df = df.merge(cl, on="id").query("cls == @SCOPE").reset_index(drop=True)
        print(f"scope-filtered to {SCOPE}: {len(df)} rows", flush=True)
    qm_mean = df[QM].mean().values; qm_std = df[QM].std().replace(0, 1).values
    print("building", len(df), "graphs", flush=True)
    graphs = []; t0 = time.time()
    qmz = ((df[QM].values - qm_mean) / qm_std)
    for k in range(len(df)):
        g = mol_to_graph(df.smiles.iloc[k], float(df.dG_orca_kcal.iloc[k]) - float(df.dG_gxtb_kcal.iloc[k]),
                         float(df.dG_gxtb_kcal.iloc[k]), qmz[k])
        if g is not None: graphs.append(g)
        if k % 20000 == 0: print(f"  {k} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"graphs": graphs, "qm_mean": qm_mean, "qm_std": qm_std}, CACHE)
    print("built+cached", len(graphs), flush=True)

N = len(graphs)
rng = np.random.default_rng(42)
idx = rng.permutation(N)
ntr, nva = int(.7 * N), int(.2 * N)
tr, va, te = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
print(f"split train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)
ys = torch.tensor([graphs[i].y.item() for i in tr]); ym, ysd = ys.mean().item(), ys.std().item()
for g in graphs: g.ystd = (g.y - ym) / ysd
def loader(ix, sh, bs=512): return DataLoader([graphs[i] for i in ix], batch_size=bs, shuffle=sh)
AD = graphs[0].x.shape[1]; BD = graphs[0].edge_attr.shape[1]; NQM = len(QM)

# ---------- generic GNN ----------
class GNN(nn.Module):
    def __init__(self, kind, h=128, L=4, heads=4, hybrid=False):
        super().__init__()
        self.kind, self.hybrid = kind, hybrid
        self.ne = nn.Linear(AD, h); self.ee = nn.Linear(BD, h)
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        for _ in range(L):
            if kind in ("gine", "gine_big"):
                mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
                c = GINEConv(mlp, edge_dim=h)
            elif kind == "gcn":
                c = GCNConv(h, h)
            elif kind == "gat":
                c = GATv2Conv(h, h // heads, heads=heads, edge_dim=h)
            elif kind == "nnconv":
                enet = nn.Sequential(nn.Linear(BD, 32), nn.ReLU(), nn.Linear(32, h * h))
                c = NNConv(h, h, enet, aggr="mean")
            self.convs.append(c); self.bns.append(nn.BatchNorm1d(h))
        hin = 2 * h + (NQM if hybrid else 0)
        self.head = nn.Sequential(nn.Linear(hin, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def forward(self, d):
        x = self.ne(d.x); e = self.ee(d.edge_attr)
        for c, bn in zip(self.convs, self.bns):
            if self.kind == "gcn":
                m = c(x, d.edge_index)
            elif self.kind == "gat":
                m = c(x, d.edge_index, e)
            elif self.kind == "nnconv":
                m = c(x, d.edge_index, d.edge_attr)
            else:
                m = c(x, d.edge_index, e)
            x = x + F.relu(bn(m))
        g = torch.cat([global_mean_pool(x, d.batch), global_add_pool(x, d.batch)], 1)
        if self.hybrid:
            g = torch.cat([g, d.qm.view(-1, NQM)], 1)
        return self.head(g).squeeze(-1)

def evaluate(model, ld):
    model.eval(); ps, ts, gs = [], [], []
    for b in ld:
        b = b.to(dev)
        with torch.no_grad(): out = model(b)
        ps.append(out.cpu()); ts.append(b.ystd.cpu()); gs.append(b.g0.cpu())
    pd_ = torch.cat(ps).numpy() * ysd + ym; td = torch.cat(ts).numpy() * ysd + ym; g0 = torch.cat(gs).numpy()
    yhat, ytrue = g0 + pd_, g0 + td
    r2 = 1 - ((ytrue - yhat) ** 2).sum() / ((ytrue - ytrue.mean()) ** 2).sum()
    return float(np.abs(ytrue - yhat).mean()), float(np.sqrt(((ytrue - yhat) ** 2).mean())), r2, ytrue, yhat

def train_one(cfg):
    torch.manual_seed(0)
    model = GNN(**{k: v for k, v in cfg.items() if k != "name"}).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    tl, vl = loader(tr, True), loader(va, False)
    nparam = sum(p.numel() for p in model.parameters())
    best, best_state, pat = 1e9, None, 0
    t0 = time.time()
    for ep in range(120):
        model.train()
        for b in tl:
            b = b.to(dev); opt.zero_grad()
            loss = F.mse_loss(model(b), b.ystd); loss.backward(); opt.step()
        vmae = evaluate(model, vl)[0]; sch.step(vmae)
        if vmae < best - 1e-4: best, best_state, pat = vmae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else: pat += 1
        if ep % 10 == 0 or pat == 0:
            print(f"  [{cfg['name']}] ep{ep:3d} val_MAE {vmae:.3f} (best {best:.3f}) {time.time()-t0:.0f}s", flush=True)
        if pat >= 15: break
    model.load_state_dict(best_state)
    mae, rmse, r2, t, p = evaluate(model, loader(te, False))
    print(f"==> {cfg['name']:12s} TEST MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}  params={nparam/1e3:.0f}k  {time.time()-t0:.0f}s", flush=True)
    return dict(name=cfg["name"], MAE=mae, RMSE=rmse, R2=r2, params=nparam), model, (t, p)

CONFIGS = [
    dict(name="gine", kind="gine", h=128, L=4),
    dict(name="gine_big", kind="gine_big", h=256, L=5),
    dict(name="gat", kind="gat", h=128, L=4, heads=4),
    dict(name="gcn", kind="gcn", h=128, L=4),
    dict(name="nnconv", kind="nnconv", h=96, L=3),
    dict(name="gine_hybrid", kind="gine", h=128, L=4, hybrid=True),
]

if ARCHS:
    want = set(ARCHS.split(","))
    CONFIGS = [c for c in CONFIGS if c["name"] in want]
    print("running only:", [c["name"] for c in CONFIGS], flush=True)
results, best_model, best_pp, best_mae, best_name = [], None, None, 1e9, None
for cfg in CONFIGS:
    try:
        r, model, pp = train_one(cfg)
        results.append(r)
        if r["MAE"] < best_mae: best_mae, best_model, best_pp, best_name = r["MAE"], model, pp, r["name"]
    except Exception as ex:
        print(f"!! {cfg['name']} FAILED: {ex}", flush=True)
        results.append(dict(name=cfg["name"], MAE=np.nan, RMSE=np.nan, R2=np.nan, params=0))

res = pd.DataFrame(results).sort_values("MAE")
res.to_csv(f"{OUT}/gnn_arch_results{SUF}.csv", index=False)
print("\n=== ARCHITECTURE COMPARISON (test, kcal/mol) ===\n" + res.to_string(index=False), flush=True)
print(f"\nbest GNN = {best_name} (MAE {best_mae:.3f})", flush=True)
import mlflow
mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow_benchmark.db"); mlflow.set_experiment("exp1_gnn_arch_full")
for _,r in res.iterrows():
    with mlflow.start_run(run_name=f"GNN_{r['name']}_full"):
        mlflow.log_params({"model":r["name"],"arch":"gnn_arch_study","n":N,"params":int(r.get("params",0))}); mlflow.log_metrics({"test_mae":float(r["MAE"]),"test_r2":float(r["R2"])})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# comparison bar
fig, ax = plt.subplots(figsize=(8, 4.2))
rr = res.dropna(subset=["MAE"])
ax.bar(rr["name"], rr["MAE"], color="#3690c0")
ax.axhline(2.46, color="g", ls="--", label="GBT Δ 2.46"); ax.axhline(4.35, color="r", ls=":", label="raw g-xTB 4.35")
for i, (_, row) in enumerate(rr.iterrows()): ax.text(i, row.MAE + .03, f"{row.MAE:.2f}", ha="center", fontsize=9)
ax.set_ylabel("test MAE vs DFT (kcal/mol)"); ax.set_title("GNN architecture study (Δ-learning g-xTB→DFT)")
ax.legend(); plt.xticks(rotation=20)
fig.tight_layout(); fig.savefig(f"{OUT}/26_gnn_arch_comparison{SUF}.png", dpi=150)
print(f"wrote 26_gnn_arch_comparison{SUF}.png", flush=True)
# best parity
t, p = best_pp
fig, ax = plt.subplots(figsize=(6, 5.6))
ax.hexbin(t, p, gridsize=70, cmap="viridis", bins="log", mincnt=1, extent=(-25, 30, -25, 30))
ax.plot([-25, 30], [-25, 30], "w--", lw=1); ax.set_xlim(-25, 30); ax.set_ylim(-25, 30)
m = res.set_index("name").loc[best_name]
ax.set_xlabel("DFT ΔG (kcal/mol)"); ax.set_ylabel(f"{best_name} g-xTB+Δ (kcal/mol)")
ax.set_title(f"Best GNN ({best_name}) vs DFT TEST\nMAE={m.MAE:.2f} RMSE={m.RMSE:.2f} R²={m.R2:.2f}")
fig.tight_layout(); fig.savefig(f"{OUT}/27_gnn_best_parity{SUF}.png", dpi=150)
print(f"wrote 27_gnn_best_parity{SUF}.png", flush=True)

# ---------- full-library prediction with best model ----------
print("predicting full-library correction with best model ...", flush=True)
pf = pd.read_csv(PROD, usecols=["id", "smiles", "dG_gxtb_kcal"] + QM, low_memory=False)
pf = pf.dropna(subset=["smiles", "dG_gxtb_kcal"]).reset_index(drop=True)
preds = np.full(len(pf), np.nan); buf, pos = [], []
qm_fill = pf[QM].values
qm_fill = np.where(np.isnan(qm_fill), qm_mean, qm_fill)
qmz = (qm_fill - qm_mean) / qm_std
best_model.eval()
def flush():
    if not buf: return
    for b in DataLoader(buf, batch_size=512, shuffle=False):
        b = b.to(dev)
        with torch.no_grad(): o = best_model(b).cpu().numpy() * ysd + ym
        for q, v in zip(pos[:len(o)], o): preds[q] = v
        pos[:len(o)] = []
    buf.clear()
for k in range(len(pf)):
    g = mol_to_graph(pf.smiles.iloc[k], 0.0, float(pf.dG_gxtb_kcal.iloc[k]), qmz[k])
    if g is not None: buf.append(g); pos.append(k)
    if len(buf) >= 20000: flush()
flush()
pf["delta_gnn"] = preds
pf["dG_gxtb_gnn_corrected"] = pf["dG_gxtb_kcal"] + preds
pf[["id", "dG_gxtb_kcal", "delta_gnn", "dG_gxtb_gnn_corrected"]].to_csv(
    f"{OUT}/products_gxtb_gnn_corrected{SUF}.csv", index=False)
print(f"wrote products_gxtb_gnn_corrected{SUF}.csv ({np.isfinite(preds).sum():,} corrected)", flush=True)
print("DONE", flush=True)
