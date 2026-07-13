#!/usr/bin/env python3
"""
Compare THREE reactant+product structure representations for predicting benzoin
ΔG directly (no xTB), on the SAME fixed 70:20:10 train/val/test split (per user
preference, see data-split-721 memory — not repeated K-fold, unlike the earlier
train_surrogate.py / train_selfies_surrogate.py CV numbers, so treat those as
only roughly comparable reference points):

  A. SELFIES bag-of-symbols   (train_selfies_surrogate.py's featurization;
                                vocab built from TRAIN split only here, unlike
                                that script's whole-dataset vocab — no leakage)
  B. Morgan/ECFP fingerprints  (radius=2, nBits=1024, reactant+product concat)
  C. Small sequence model      (shared token embedding + GRU encoder for
                                reactant and product SELFIES sequences, concat
                                pooled reps, small MLP head; early-stopped on
                                val MAE) — the only one that sees SEQUENCE
                                order/branching, not just a count or a fixed
                                circular-substructure hash.

A and B use xgb + ridge (same architecture as the rest of this project's
surrogates). C is a small PyTorch model (CPU; n~1600 is tiny, no GPU needed).

Usage
  python pipeline/train_reaction_repr_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc
from train_selfies_surrogate import (
    make_benzoin_smiles, selfies_tokens, build_vocab, count_vector,
)

RDLogger.DisableLog("rdApp.*")
SEED = 42


def metrics(y_true, y_pred) -> dict:
    return dict(MAE=mean_absolute_error(y_true, y_pred),
                RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
                R2=r2_score(y_true, y_pred))


def fixed_721_split(n: int, seed: int = SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_tr = int(round(0.70 * n))
    n_val = int(round(0.20 * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_val], idx[n_tr + n_val:]


def load_pairs():
    """Joint aromatic training table + benzoin product SMILES per row."""
    tbl = dc.load_training_table()
    df = tbl.df.reset_index(drop=True)
    bz = [make_benzoin_smiles(s) for s in df["SMILES"]]
    ok = []
    for s, b in zip(df["SMILES"], bz):
        ok.append(b is not None and selfies_tokens(s) is not None and selfies_tokens(b) is not None)
    keep = np.array(ok)
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"  dropped {n_drop}/{len(df)} rows (benzoin SMARTS or SELFIES encoding failed)")
    react_smi = df["SMILES"].to_numpy()[keep]
    prod_smi = np.array(bz)[keep]
    y = tbl.dG_dft[keep]
    dG_xtb = tbl.dG_xtb[keep]
    return react_smi, prod_smi, y, dG_xtb


# ---------------------------------------------------------------- A: SELFIES
def featurize_selfies(react_smi, prod_smi, tr_idx, min_count=2):
    react_tok = [selfies_tokens(s) for s in react_smi]
    prod_tok = [selfies_tokens(s) for s in prod_smi]
    vocab = build_vocab([react_tok[i] for i in tr_idx] + [prod_tok[i] for i in tr_idx], min_count)
    Xr = np.stack([count_vector(t, vocab) for t in react_tok])
    Xp = np.stack([count_vector(t, vocab) for t in prod_tok])
    len_r = np.array([[len(t)] for t in react_tok], dtype=float)
    len_p = np.array([[len(t)] for t in prod_tok], dtype=float)
    return np.hstack([Xr, Xp, len_r, len_p]), len(vocab)


# ------------------------------------------------------------------- B: ECFP
def featurize_ecfp(react_smi, prod_smi, radius=2, n_bits=1024):
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def fp(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(n_bits, dtype=float)
        return np.array(gen.GetFingerprint(mol), dtype=float)

    Xr = np.stack([fp(s) for s in react_smi])
    Xp = np.stack([fp(s) for s in prod_smi])
    return np.hstack([Xr, Xp])


def xgb_ridge_eval(name, X, y, tr, va, te, dG_xtb_te):
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "xgb": XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                             random_state=SEED, n_jobs=-1),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }
    out = {}
    for kind, m in models.items():
        m.fit(X[tr], y[tr])
        val_m = metrics(y[va], m.predict(X[va]))
        test_m = metrics(y[te], m.predict(X[te]))
        out[f"{name}-{kind}"] = (val_m, test_m)
    return out


# ---------------------------------------------------------- C: sequence model
def train_seq_model(react_smi, prod_smi, y, tr, va, te):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    react_tok = [selfies_tokens(s) for s in react_smi]
    prod_tok = [selfies_tokens(s) for s in prod_smi]
    train_toks = [react_tok[i] for i in tr] + [prod_tok[i] for i in tr]
    vocab = sorted({tok for toks in train_toks for tok in toks})
    tok2id = {tok: i + 2 for i, tok in enumerate(vocab)}   # 0=PAD, 1=UNK
    PAD, UNK = 0, 1
    maxlen = max(max(len(t) for t in react_tok), max(len(t) for t in prod_tok))
    maxlen = min(maxlen, 80)

    def encode(toks):
        ids = [tok2id.get(t, UNK) for t in toks[:maxlen]]
        ids = ids + [PAD] * (maxlen - len(ids))
        return ids

    Rr = torch.tensor([encode(t) for t in react_tok], dtype=torch.long)
    Rp = torch.tensor([encode(t) for t in prod_tok], dtype=torch.long)
    Y = torch.tensor(y, dtype=torch.float32)

    class SeqEncoder(nn.Module):
        def __init__(self, vocab_size, emb_dim=32, hid_dim=64):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
            self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(2 * hid_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )

        def encode_one(self, x):
            e = self.emb(x)
            _, h = self.gru(e)
            return h.squeeze(0)

        def forward(self, xr, xp):
            hr = self.encode_one(xr)
            hp = self.encode_one(xp)
            return self.head(torch.cat([hr, hp], dim=-1)).squeeze(-1)

    model = SeqEncoder(vocab_size=len(vocab) + 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    tr_t, va_t, te_t = map(lambda a: torch.as_tensor(a), (tr, va, te))
    best_val, best_state, patience, bad = float("inf"), None, 25, 0
    batch_size = 64
    n_tr = len(tr)
    for epoch in range(300):
        model.train()
        perm = tr_t[torch.randperm(n_tr)]
        for i in range(0, n_tr, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Rr[b], Rp[b])
            loss = loss_fn(pred, Y[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Rr[va_t], Rp[va_t])
            val_mae = (val_pred - Y[va_t]).abs().mean().item()
        if val_mae < best_val - 1e-4:
            best_val, best_state, bad = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(Rr[va_t], Rp[va_t]).numpy()
        test_pred = model(Rr[te_t], Rp[te_t]).numpy()
    return (metrics(y[va], val_pred), metrics(y[te], test_pred)), epoch + 1


def main() -> int:
    react_smi, prod_smi, y, dG_xtb = load_pairs()
    n = len(y)
    tr, va, te = fixed_721_split(n)
    print(f"n={n}  split train/val/test = {len(tr)}/{len(va)}/{len(te)} (70:20:10, seed={SEED})\n")

    base_te = metrics(y[te], dG_xtb[te])
    print(f"  [ref] xTB-only (no ML), test split   MAE {base_te['MAE']:.3f}  "
          f"RMSE {base_te['RMSE']:.3f}  R2 {base_te['R2']:.3f}")
    print("  [ref] RDKit-2D surrogate (xgb, CV)    MAE ~2.92  (train_surrogate.py, different protocol)")
    print("  [ref] Delta-model floor (xgb, CV)     MAE ~2.19  (modeling-direction memory, different protocol)\n")

    results = {}

    print("== A. SELFIES bag-of-symbols ==")
    Xa, vsz = featurize_selfies(react_smi, prod_smi, tr)
    print(f"  vocab={vsz} symbols, {Xa.shape[1]} features")
    results.update(xgb_ridge_eval("selfies", Xa, y, tr, va, te, dG_xtb[te]))

    print("\n== B. Morgan/ECFP fingerprints (r=2, 1024 bits x2) ==")
    Xb = featurize_ecfp(react_smi, prod_smi)
    print(f"  {Xb.shape[1]} features")
    results.update(xgb_ridge_eval("ecfp", Xb, y, tr, va, te, dG_xtb[te]))

    print("\n== C. Small sequence model (embedding + GRU, reactant+product) ==")
    (val_m, test_m), n_epochs = train_seq_model(react_smi, prod_smi, y, tr, va, te)
    results["seq-gru"] = (val_m, test_m)
    print(f"  stopped at epoch {n_epochs}")

    print("\n" + "=" * 70)
    print(f"{'method':<16}{'val MAE':>10}{'test MAE':>10}{'test RMSE':>11}{'test R2':>9}")
    for name, (val_m, test_m) in results.items():
        print(f"{name:<16}{val_m['MAE']:>10.3f}{test_m['MAE']:>10.3f}"
              f"{test_m['RMSE']:>11.3f}{test_m['R2']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
