#!/usr/bin/env python
"""Pure-SMILES 2D GINE GNN predicting REAL DFT (r2SCAN-3c) benzoin dG_orca_kcal
from the aldehyde reactant graph alone -- no QM/xTB descriptors at all.

Follow-up to data/raw/screen_v6/gnn_dG.py (which predicted the cheap xTB proxy
label over aromatic-only, n=146,741) and pipeline/train_reaction_repr_compare.py
(SELFIES/ECFP/seq surrogates, DFT label but n=1,633 only). This run uses the now-
available REAL DFT label at FULL library scale (~219k, ALL cho_class categories
per aromatic-only-scope being superseded) -- the natural next step once DFT-SP
labels landed for ~the whole library.

Same seed=42 random 70:20:10 holdout protocol as the tabular/GNN champion scripts
(data-split-721) for direct MAE comparability against MORDREDSLIM271_BDEGXTB
(1.503) and the GNN+tabular blend (1.427) -- NOT scaffold-split, unlike the older
xTB-label script, specifically so this number is apples-to-apples with those.
"""
import os, time, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"
OUT = f"{R}/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
CACHE = f"{OUT}/gnn_cache_smiles_dft_full.pt"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, flush=True)

# ---------- featurizer (same scheme as gnn_dG.py) ----------
ELEMS = ['B','C','N','O','F','Si','P','S','Cl','Se','Br','I']
def onehot(x, choices):
    return [int(x == c) for c in choices] + [int(x not in choices)]
def atom_feat(a):
    return (onehot(a.GetSymbol(), ELEMS)
            + onehot(a.GetTotalDegree(), [0,1,2,3,4,5])
            + onehot(a.GetFormalCharge(), [-2,-1,0,1,2])
            + onehot(str(a.GetHybridization()), ['SP','SP2','SP3','SP3D','SP3D2'])
            + onehot(a.GetTotalNumHs(), [0,1,2,3,4])
            + [int(a.GetIsAromatic()), int(a.IsInRing())])
BT = {Chem.BondType.SINGLE:0, Chem.BondType.DOUBLE:1, Chem.BondType.TRIPLE:2, Chem.BondType.AROMATIC:3}
def bond_feat(b):
    bt = [0,0,0,0]; bt[BT.get(b.GetBondType(),0)] = 1
    return bt + [int(b.GetIsConjugated()), int(b.IsInRing())]

def mol_to_graph(smi, y):
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0: return None
    x = torch.tensor([atom_feat(a) for a in m.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); f = bond_feat(b)
        ei += [[i,j],[j,i]]; ea += [f,f]
    if not ei:
        ei = [[0,0]]; ea = [[0,0,0,0,0,0]]
    return Data(x=x, edge_index=torch.tensor(ei).t().contiguous(),
                edge_attr=torch.tensor(ea, dtype=torch.float),
                y=torch.tensor([y], dtype=torch.float), smi=smi)

# ---------- load + build (cached) ----------
if os.path.exists(CACHE):
    blob = torch.load(CACHE, weights_only=False); graphs = blob["graphs"]
    print("loaded cache", len(graphs), flush=True)
else:
    df = pd.read_parquet(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet",
                          columns=["id", "smiles", "dG_orca_kcal"])
    df = df.dropna(subset=["dG_orca_kcal", "smiles"])
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)  # same loose sanity bound as the tabular champion
    print("building", len(df), "graphs (ALL cho_class, real DFT label)", flush=True)
    graphs = []
    t0 = time.time()
    for k, (smi, yv) in enumerate(zip(df["smiles"], df["dG_orca_kcal"])):
        g = mol_to_graph(smi, float(yv))
        if g is None: continue
        graphs.append(g)
        if k % 20000 == 0: print(f"  {k} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"graphs": graphs}, CACHE)
    print("built+cached", len(graphs), flush=True)

N = len(graphs)

# ---------- seed=42 random 70:20:10 split (matches the tabular/GNN champion protocol) ----------
rng = np.random.default_rng(42)
order = rng.permutation(N)
n_te, n_va = int(0.10 * N), int(0.20 * N)
te = order[:n_te]; va = order[n_te:n_te+n_va]; tr = order[n_te+n_va:]
print(f"split train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)

ys = torch.tensor([graphs[i].y.item() for i in tr])
ym, ysd = ys.mean().item(), ys.std().item()
for g in graphs: g.y = (g.y - ym) / ysd

def loader(idx, shuf): return DataLoader([graphs[i] for i in idx], batch_size=256, shuffle=shuf)
tl, vl, sl = loader(tr, True), loader(va, False), loader(te, False)

# ---------- GINE model (identical arch to gnn_dG.py) ----------
class GNN(nn.Module):
    def __init__(self, ad, bd, h=128, L=4):
        super().__init__()
        self.ne = nn.Linear(ad, h); self.ee = nn.Linear(bd, h)
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        for _ in range(L):
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.convs.append(GINEConv(mlp, edge_dim=h)); self.bns.append(nn.BatchNorm1d(h))
        self.head = nn.Sequential(nn.Linear(2*h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))
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
    model.train(train); ps, ts = [], []
    tot = 0.0
    for b in ld:
        b = b.to(dev)
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(b); loss = F.mse_loss(out, b.y)
            if train: loss.backward(); opt.step()
        tot += loss.item() * b.num_graphs
        ps.append(out.detach().cpu()); ts.append(b.y.detach().cpu())
    p = torch.cat(ps).numpy() * ysd + ym; t = torch.cat(ts).numpy() * ysd + ym
    ss = 1 - ((t-p)**2).sum() / ((t-t.mean())**2).sum()
    return tot/len(ld.dataset), ss, float(np.sqrt(((t-p)**2).mean())), float(np.abs(t-p).mean()), t, p

train_losses, val_r2s = [], []
best_v, best_state, patience = -1e9, None, 0
for ep in range(150):
    tr_loss, tr_r2, _, tr_mae, _, _ = run(tl, True)
    _, vr2, _, v_mae, _, _ = run(vl, False)
    train_losses.append(tr_loss); val_r2s.append(vr2)
    sched.step(-vr2)
    if vr2 > best_v: best_v, best_state, patience = vr2, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    else: patience += 1
    if ep % 5 == 0 or patience == 0:
        print(f"ep{ep:3d} train_MAE {tr_mae:.3f} val_MAE {v_mae:.3f} val_R2 {vr2:.3f} (best {best_v:.3f})", flush=True)
    if patience >= 15: print("early stop", flush=True); break

model.load_state_dict(best_state)
_, r2, rmse, mae, t, p = run(sl, False)
print(f"\nTEST  n={len(te)}  R2={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}  (val {best_v:.3f})", flush=True)
print(f"reference: tabular champion MORDREDSLIM271_BDEGXTB MAE=1.503, GNN+tabular blend MAE=1.427", flush=True)

# ---------- diagnostics (training-runs-full-diagnostics: keep curves + parity + residual) ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.hexbin(t, p, gridsize=70, cmap="viridis", bins="log", mincnt=1)
lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
ax.plot([lo, hi], [lo, hi], "r--", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("actual DFT (r2SCAN-3c) dG_orca (kcal/mol)"); ax.set_ylabel("GNN predicted dG (kcal/mol)")
ax.set_title(f"pure-SMILES GINE, full library (n={N:,}) — TEST\nR²={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gnn_smiles_dft_full_parity_20260713.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
resid = p - t
ax.hist(resid, bins=100, color="#2171b5", alpha=0.8)
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("prediction - DFT (kcal/mol)"); ax.set_ylabel("count")
ax.set_title("pure-SMILES GINE residual distribution — TEST")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gnn_smiles_dft_full_residual_20260713.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(train_losses, label="train loss (MSE, standardized)")
ax2 = ax.twinx(); ax2.plot(val_r2s, color="#cb181d", label="val R2")
ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax2.set_ylabel("val R2", color="#cb181d")
lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc="upper right")
ax.set_title("pure-SMILES GINE training curves")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gnn_smiles_dft_full_traincurve_20260713.png", dpi=150); plt.close(fig)

pd.DataFrame({"dG_true": t, "dG_pred": p}).to_csv(f"{OUT}/gnn_smiles_dft_full_test_predictions_20260713.csv", index=False)
with open(f"{OUT}/REPORT_gnn_smiles_dft_full_20260713.md", "w") as f:
    f.write(f"""# Pure-SMILES GINE on full-library REAL DFT labels (2026-07-13)

Reactant (aldehyde) SMILES only, no QM/xTB descriptors. Target: `dG_orca_kcal`
(real DFT r2SCAN-3c label, `data/raw/dft_sp_funnelv3/dft_labels_all.parquet`),
ALL cho_class categories, seed=42 random 70:20:10 split (same protocol as the
tabular/GNN champion).

- N total = {N:,}, train = {len(tr):,}, val = {len(va):,}, test = {len(te):,}
- **Test MAE = {mae:.3f} kcal/mol, RMSE = {rmse:.3f}, R2 = {r2:.3f}**
- Reference: tabular champion MORDREDSLIM271_BDEGXTB (72 QM + 199 mordred + 4
  BDE/BDFE) test MAE = 1.503; GNN+tabular blend test MAE = 1.427.
- Prior pure-SMILES attempts (n=1,633 only): SELFIES counts 3.16, ECFP+xgb 3.01,
  seq-GRU 3.25, vs RDKit-2D descriptor surrogate ~2.92 (all far short of the
  Δ-learning tabular floor).

Params: {sum(p.numel() for p in model.parameters()):,}. Full diagnostics: parity/
residual/training-curve PNGs + per-molecule test predictions CSV in this dir.
""")
print("wrote report + figs + predictions csv", flush=True)
