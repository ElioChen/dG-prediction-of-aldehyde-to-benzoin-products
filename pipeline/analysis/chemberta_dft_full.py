#!/usr/bin/env python
"""Fine-tune pretrained ChemBERTa (seyonec/ChemBERTa-zinc-base-v1, RoBERTa MLM
pretrained on ~10M ZINC SMILES) on the aldehyde reactant SMILES -> real DFT
(r2SCAN-3c) benzoin dG_orca_kcal task, at full library scale.

Third pure-SMILES angle after gnn_smiles_dft_full.py (from-scratch GINE graph,
same task/data) and the earlier tiny-n (1,633) SELFIES/ECFP/seq-GRU surrogates
(pipeline/train_reaction_repr_compare.py) -- this one tests whether transfer
learning from a much larger *unlabeled* chemical corpus helps a from-scratch
model beat, given our real bottleneck was always data volume for the
task-specific label, not architecture.

Same seed=42 random 70:20:10 split protocol as the tabular/GNN champion +
gnn_smiles_dft_full.py, for direct MAE comparability.
"""
import os, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

R = "/scratch-shared/schen3/benzoin-dg"
OUT = f"{R}/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
CKPT = "seyonec/ChemBERTa-zinc-base-v1"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, flush=True)

# ---------- data (identical source/filter to gnn_smiles_dft_full.py) ----------
df = pd.read_parquet(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet",
                      columns=["id", "smiles", "dG_orca_kcal"])
df = df.dropna(subset=["dG_orca_kcal", "smiles"])
df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
N = len(df)
print(f"N = {N:,}", flush=True)

rng = np.random.default_rng(42)
order = rng.permutation(N)
n_te, n_va = int(0.10 * N), int(0.20 * N)
te_i = order[:n_te]; va_i = order[n_te:n_te+n_va]; tr_i = order[n_te+n_va:]
print(f"split train {len(tr_i)} / val {len(va_i)} / test {len(te_i)}", flush=True)

y = df["dG_orca_kcal"].values.astype(np.float32)
ym, ysd = y[tr_i].mean(), y[tr_i].std()

tokenizer = AutoTokenizer.from_pretrained(CKPT)

class SmilesDS(Dataset):
    def __init__(self, idx):
        self.smi = df["smiles"].values[idx]; self.y = (y[idx] - ym) / ysd
    def __len__(self): return len(self.smi)
    def __getitem__(self, i): return self.smi[i], self.y[i]

def collate(batch):
    smis, ys = zip(*batch)
    enc = tokenizer(list(smis), return_tensors="pt", padding=True, truncation=True, max_length=128)
    return enc, torch.tensor(ys, dtype=torch.float32)

tr_ds, va_ds, te_ds = SmilesDS(tr_i), SmilesDS(va_i), SmilesDS(te_i)
tl = DataLoader(tr_ds, batch_size=64, shuffle=True, collate_fn=collate)
vl = DataLoader(va_ds, batch_size=128, shuffle=False, collate_fn=collate)
sl = DataLoader(te_ds, batch_size=128, shuffle=False, collate_fn=collate)

class ChemBERTaRegressor(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(ckpt)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h//2, 1))
    def forward(self, enc):
        out = self.backbone(**enc)
        cls = out.last_hidden_state[:, 0]  # [CLS]-equivalent first token
        return self.head(cls).squeeze(-1)

model = ChemBERTaRegressor(CKPT).to(dev)
print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
loss_fn = nn.MSELoss()

def run(ld, train):
    model.train(train); tot = 0.0; ps, ts = [], []
    for enc, yb in ld:
        enc = {k: v.to(dev) for k, v in enc.items()}; yb = yb.to(dev)
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(enc); loss = loss_fn(out, yb)
            if train: loss.backward(); opt.step()
        tot += loss.item() * len(yb)
        ps.append(out.detach().cpu()); ts.append(yb.detach().cpu())
    p = torch.cat(ps).numpy() * ysd + ym; t = torch.cat(ts).numpy() * ysd + ym
    ss = 1 - ((t-p)**2).sum() / ((t-t.mean())**2).sum()
    return tot/len(ld.dataset), ss, float(np.sqrt(((t-p)**2).mean())), float(np.abs(t-p).mean()), t, p

train_losses, val_r2s = [], []
best_v, best_state, patience = -1e9, None, 0
MAX_EPOCHS = 12  # fine-tuning a pretrained transformer needs far fewer epochs than from-scratch GINE
for ep in range(MAX_EPOCHS):
    t0 = time.time()
    tr_loss, _, _, tr_mae, _, _ = run(tl, True)
    _, vr2, _, v_mae, _, _ = run(vl, False)
    train_losses.append(tr_loss); val_r2s.append(vr2)
    sched.step(-vr2)
    if vr2 > best_v: best_v, best_state, patience = vr2, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    else: patience += 1
    print(f"ep{ep:2d} train_MAE {tr_mae:.3f} val_MAE {v_mae:.3f} val_R2 {vr2:.3f} (best {best_v:.3f}) {time.time()-t0:.0f}s", flush=True)
    if patience >= 3: print("early stop", flush=True); break

model.load_state_dict(best_state)
_, r2, rmse, mae, t, p = run(sl, False)
print(f"\nTEST  n={len(te_i)}  R2={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}  (val {best_v:.3f})", flush=True)
print(f"reference: tabular champion MAE=1.503, GNN+tabular blend MAE=1.427, from-scratch GINE (this same task) see fig_gnn_smiles_dft_full", flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.hexbin(t, p, gridsize=70, cmap="viridis", bins="log", mincnt=1)
lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
ax.plot([lo, hi], [lo, hi], "r--", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("actual DFT (r2SCAN-3c) dG_orca (kcal/mol)"); ax.set_ylabel("ChemBERTa predicted dG (kcal/mol)")
ax.set_title(f"fine-tuned ChemBERTa, full library (n={N:,}) — TEST\nR²={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_chemberta_dft_full_parity_20260713.png", dpi=150); plt.close(fig)

pd.DataFrame({"dG_true": t, "dG_pred": p}).to_csv(f"{OUT}/chemberta_dft_full_test_predictions_20260713.csv", index=False)
with open(f"{OUT}/REPORT_chemberta_dft_full_20260713.md", "w") as f:
    f.write(f"""# Fine-tuned ChemBERTa on full-library REAL DFT labels (2026-07-13)

Pretrained checkpoint: `{CKPT}` (RoBERTa MLM, ~10M ZINC SMILES). Fine-tuned
end-to-end (not frozen-backbone) with a small regression head on the [CLS] token,
on the SAME task/data/split as `gnn_smiles_dft_full.py`: reactant (aldehyde)
SMILES only, target `dG_orca_kcal` (real DFT r2SCAN-3c), ALL cho_class
categories, seed=42 random 70:20:10 split.

- N total = {N:,}, train = {len(tr_i):,}, val = {len(va_i):,}, test = {len(te_i):,}
- **Test MAE = {mae:.3f} kcal/mol, RMSE = {rmse:.3f}, R2 = {r2:.3f}**
- Reference: tabular champion MORDREDSLIM271_BDEGXTB MAE = 1.503; GNN+tabular
  blend MAE = 1.427; prior tiny-n (1,633) pure-SMILES attempts: SELFIES 3.16,
  ECFP+xgb 3.01, seq-GRU 3.25, RDKit-2D 2.92.

Params: {sum(p.numel() for p in model.parameters()):,} ({MAX_EPOCHS} max epochs,
early-stopped on val R2, patience 3 -- pretrained transformers converge much
faster than from-scratch GNNs). Full diagnostics: parity PNG + per-molecule test
predictions CSV in this dir.
""")
print("wrote report + fig + predictions csv", flush=True)
print("DONE", flush=True)
