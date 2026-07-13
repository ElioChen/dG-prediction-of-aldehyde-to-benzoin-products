#!/usr/bin/env python3
"""
SELFIES-based 2D surrogate for benzoin ΔG — same "fast, no-quantum" tier as
train_surrogate.py, but the feature representation is a bag-of-SELFIES-symbols
over BOTH the aldehyde (reactant) and its self-condensation product (benzoin),
instead of RDKit-2D descriptors on the aldehyde alone.

Reaction: 2 R-CHO -> R-CH(OH)-C(=O)-R  (thermo_orca._BENZOIN_RXN SMARTS,
reimplemented here so this script has no dependency on the xTB/ORCA-calling
parts of thermo_orca.py). Target = dG_orca_kcal DIRECT (not the xTB residual),
same as train_surrogate.py, so the two are directly comparable.

Why reactant+product jointly: modeling-direction memory found Δ-model error
grows with BENZOIN-PRODUCT flexibility, not aldehyde size alone — the product
structure (which the reaction SMARTS deterministically fixes from the
reactant) may carry signal the reactant-only 2D surrogate can't see.

Featurization: SELFIES-encode both SMILES, split into symbol tokens, and count
each token against a shared vocabulary built from the training set (bag-of-
symbols, analogous to a fragment count fingerprint but over SELFIES grammar
tokens instead of substructures). Reactant and product get separate count
blocks over the SAME vocabulary so the model can weight "this ring symbol on
the reactant side" differently from "on the product side". Token length of
each string is added as two scalar features (crude size/flexibility proxy).

This is a first, cheap experiment (bag-of-symbols + xgb/ridge) — no sequence
model. If it beats or matches the RDKit-2D surrogate, a learned embedding
(small MLP/RNN in envs/gnn, which has torch) would be the natural next step.

Usage
  python pipeline/train_selfies_surrogate.py
  python pipeline/train_selfies_surrogate.py --min-count 2 --kinds xgb ridge
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import RepeatedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc

RDLogger.DisableLog("rdApp.*")

# Single source of truth for this SMARTS is thermo_orca._BENZOIN_RXN
# (pipeline/compute/thermo_orca.py:80-83) — copied here to avoid pulling in
# that module's xtb/orca/conf_funnel_v3 dependencies for a pure-SMILES script.
_BENZOIN_RXN = AllChem.ReactionFromSmarts(
    "[CX3H1:1](=[O:2])[#6:5].[CX3H1:3](=[O:4])[#6:6]"
    ">>[C:5][C:1]([OH1:2])[C:3](=[O:4])[C:6]"
)


def make_benzoin_smiles(ald_smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(ald_smiles)
    if mol is None:
        return None
    try:
        products = _BENZOIN_RXN.RunReactants((mol, mol))
    except Exception:
        return None
    for prod_tuple in products:
        try:
            p = prod_tuple[0]
            Chem.SanitizeMol(p)
            return Chem.MolToSmiles(p)
        except Exception:
            continue
    return None


def selfies_tokens(smiles: str) -> list[str] | None:
    try:
        enc = sf.encoder(smiles)
    except Exception:
        return None
    return list(sf.split_selfies(enc))


def build_vocab(token_lists: list[list[str]], min_count: int) -> dict[str, int]:
    counts = Counter(tok for toks in token_lists for tok in toks)
    vocab = sorted(tok for tok, c in counts.items() if c >= min_count)
    return {tok: i for i, tok in enumerate(vocab)}


def count_vector(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=float)
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            v[idx] += 1.0
    return v


def build_features(df: pd.DataFrame, min_count: int):
    react_tok, prod_tok, keep = [], [], []
    for smi in df["SMILES"]:
        bz = make_benzoin_smiles(smi)
        rt = selfies_tokens(smi) if bz else None
        pt = selfies_tokens(bz) if bz else None
        if rt and pt:
            react_tok.append(rt); prod_tok.append(pt); keep.append(True)
        else:
            react_tok.append([]); prod_tok.append([]); keep.append(False)
    keep = np.array(keep)
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"  dropped {n_drop}/{len(df)} rows (benzoin SMARTS or SELFIES encoding failed)")

    react_tok = [t for t, k in zip(react_tok, keep) if k]
    prod_tok = [t for t, k in zip(prod_tok, keep) if k]
    vocab = build_vocab(react_tok + prod_tok, min_count)
    print(f"  SELFIES vocab: {len(vocab)} symbols (min_count={min_count})")

    Xr = np.stack([count_vector(t, vocab) for t in react_tok])
    Xp = np.stack([count_vector(t, vocab) for t in prod_tok])
    len_r = np.array([[len(t)] for t in react_tok], dtype=float)
    len_p = np.array([[len(t)] for t in prod_tok], dtype=float)
    X = np.hstack([Xr, Xp, len_r, len_p])
    names = ([f"react::{s}" for s in vocab] + [f"prod::{s}" for s in vocab]
              + ["react_len", "prod_len"])
    return X, names, keep


def factory(kind, seed):
    if kind == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                             random_state=seed, n_jobs=-1)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if kind == "gpr":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        k = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=k, alpha=1e-6, normalize_y=True))
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--kinds", nargs="+", default=["xgb", "ridge"])
    ap.add_argument("--min-count", type=int, default=2,
                     help="drop SELFIES symbols seen fewer than this many times")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    kw = {"parquet": a.parquet} if a.parquet else {}
    tbl = dc.load_training_table(**kw)
    df = tbl.df.reset_index(drop=True)

    X, names, keep = build_features(df, a.min_count)
    y_direct = tbl.dG_dft[keep]
    dG_xtb = tbl.dG_xtb[keep]
    print(f"\nSELFIES surrogate: n={len(y_direct)} molecules, {X.shape[1]} features "
          f"(reactant+product bag-of-symbols + 2 length scalars), target=dG_orca (direct)\n")

    base = dc.metrics_vs_dft(y_direct, dG_xtb)
    print(f"  [ref] xTB-only (no ML)         MAE {base['MAE']:.3f}  RMSE {base['RMSE']:.3f}")
    print(f"  [ref] RDKit-2D surrogate (xgb)  MAE ~2.92  (from train_surrogate.py, same scope)")
    print(f"  [ref] Delta-model floor (xgb)   MAE ~2.19  (from modeling-direction memory)\n")

    rkf = RepeatedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=a.seed)
    for kind in a.kinds:
        maes, rmses = [], []
        for tr, te in rkf.split(X):
            m = factory(kind, a.seed)
            m.fit(X[tr], y_direct[tr])
            p = m.predict(X[te])
            d = np.abs(p - y_direct[te])
            maes.append(d.mean()); rmses.append(np.sqrt((d ** 2).mean()))
        print(f"  SELFIES-direct {kind:6s}  MAE {np.mean(maes):.3f}±{np.std(maes):.3f}  "
              f"RMSE {np.mean(rmses):.3f}")

    if "xgb" in a.kinds:
        m = factory("xgb", a.seed)
        m.fit(X, y_direct)
        imp = m.feature_importances_
        top = np.argsort(imp)[::-1][:15]
        print("\n  top-15 xgb feature importances (full-data fit, for inspection only):")
        for i in top:
            print(f"    {names[i]:20s} {imp[i]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
