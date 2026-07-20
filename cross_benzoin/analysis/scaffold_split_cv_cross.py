#!/usr/bin/env python3
"""
Scaffold-split generalization test for the cross-benzoin Delta-model --
mirrors homo's pipeline/analysis/exp_scaffold_split.py (never done for
cross). Reports the "generalization gap": random-split test MAE vs
scaffold-disjoint-split test MAE, on the SAME held-out fraction, same
architecture as the current champion (single-XGB, 260-feat all_raw+mordred).

Unlike homo (A+A, one molecule per row), cross pairs have TWO molecules
(donor+acceptor), so a naive per-molecule scaffold split could leak: the
same scaffold could appear as a donor in train and an acceptor in test.
Fixed via union-find over (donor_scaffold, acceptor_scaffold) co-occurrence
-- any two scaffolds that ever co-occur in the same pair are merged into one
component, and whole components (not just whole scaffolds) are held out for
test, guaranteeing zero scaffold leakage in either role.

Usage:
  python cross_benzoin/analysis/scaffold_split_cv_cross.py \
      --table data/cross_benzoin/cross_round6/cross_train_table_6rounds_mordred_slim120_matched.parquet \
      --outdir data/cross_benzoin/cross_round6/scaffold_split_check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "cross_benzoin"))
import delta_core as dc  # noqa: E402
from train_cross_delta import _feature_blocks, TARGET_COL, BASELINE_COL  # noqa: E402


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"__invalid__{smiles}"
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaf) if scaf is not None else ""
        return s if s else f"__no_ring__{smiles}"
    except Exception:
        return f"__error__{smiles}"


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def fit_eval(df: pd.DataFrame, feats: list[str], tr_idx: np.ndarray, te_idx: np.ndarray, seed: int) -> dict:
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = Xdf.iloc[tr_idx].median(numeric_only=True)
    X = Xdf.fillna(medians)
    y = (df[TARGET_COL] - df[BASELINE_COL]).to_numpy()
    m = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, seed)
    m.fit(X.iloc[tr_idx], y[tr_idx])
    pred = df[BASELINE_COL].to_numpy()[te_idx] + m.predict(X.iloc[te_idx])
    return dc.metrics_vs_dft(df[TARGET_COL].to_numpy()[te_idx], pred)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--n-repeats", type=int, default=5,
                     help="repeat both splits with different seeds for a less noisy estimate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.table).reset_index(drop=True)
    print(f"loaded {len(df)} rows, {df['pair_key'].nunique()} unique pairs")

    print("computing Murcko scaffolds for donor+acceptor...")
    donor_scaf = df["donor_smiles"].map(scaffold)
    acceptor_scaf = df["acceptor_smiles"].map(scaffold)

    uf = UnionFind()
    for d, a in zip(donor_scaf, acceptor_scaf):
        uf.union(d, a)
    df["scaffold_component"] = [uf.find(d) for d in donor_scaf]
    n_components = df["scaffold_component"].nunique()
    comp_sizes = df["scaffold_component"].value_counts()
    print(f"{n_components} connected scaffold-components (vs "
          f"{donor_scaf.nunique()} raw donor scaffolds, {acceptor_scaf.nunique()} raw acceptor scaffolds)")
    print(f"largest component: {comp_sizes.iloc[0]} rows "
          f"({comp_sizes.iloc[0] / len(df):.1%} of all data) -- "
          f"structural finding: scaffold-pairing graph has one giant component, "
          f"can never be placed in test (would consume the whole test budget alone)")
    giant_component = comp_sizes.index[0]

    all_feats = [c for c in df.columns if c not in {
        "id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
        "donor_smiles", "acceptor_smiles", "smiles",
        "dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal", "scaffold_component"}]
    feats = _feature_blocks(all_feats)["all_raw_blocks+mordred"]
    print(f"{len(feats)} features (all_raw_blocks+mordred, matching current champion)")

    rng = np.random.default_rng(args.seed)
    # the giant component can NEVER be placed in test (it alone dwarfs any sane test budget) --
    # it always stays in train; test can only be assembled from the remaining small components.
    non_giant_components = [c for c in df["scaffold_component"].unique() if c != giant_component]
    comp_to_idx = df.groupby("scaffold_component").indices
    max_achievable_test = len(df) - comp_sizes.loc[giant_component]
    print(f"max achievable clean scaffold-disjoint test set (excludes giant component): "
          f"{max_achievable_test} rows ({max_achievable_test / len(df):.1%}) -- "
          f"this caps the real test-fraction achievable, {args.test_frac:.0%} requested may not be reachable")

    rows = []
    for rep in range(args.n_repeats):
        n = len(df)
        n_test_target = min(int(round(args.test_frac * n)), max_achievable_test)

        # --- scaffold-disjoint split: accumulate SMALL components first (deterministic-ish,
        # shuffled order among non-giant components only) until target reached ---
        comp_order = rng.permutation(non_giant_components)
        te_scaf = []
        for c in comp_order:
            if len(te_scaf) >= n_test_target:
                break
            te_scaf.extend(comp_to_idx[c].tolist())
        te_scaf = np.array(te_scaf)
        tr_scaf = np.setdiff1d(np.arange(n), te_scaf)
        m_scaf = fit_eval(df, feats, tr_scaf, te_scaf, args.seed + rep)

        # --- random split baseline: SAME achieved n_test, for a fair matched comparison ---
        perm = rng.permutation(n)
        te_rand, tr_rand = perm[:len(te_scaf)], perm[len(te_scaf):]
        m_rand = fit_eval(df, feats, tr_rand, te_rand, args.seed + rep)

        gap = m_scaf["MAE"] - m_rand["MAE"]
        print(f"[rep {rep}] random n_test={len(te_rand)} MAE={m_rand['MAE']:.3f} R2={m_rand['R2']:.3f}  |  "
              f"scaffold n_test={len(te_scaf)} n_train={len(tr_scaf)} MAE={m_scaf['MAE']:.3f} R2={m_scaf['R2']:.3f}  |  gap={gap:+.3f}")
        rows.append({"rep": rep, "random_MAE": m_rand["MAE"], "random_R2": m_rand["R2"],
                     "random_n_test": len(te_rand),
                     "scaffold_MAE": m_scaf["MAE"], "scaffold_R2": m_scaf["R2"],
                     "scaffold_n_test": len(te_scaf), "scaffold_n_train": len(tr_scaf), "gap": gap})

    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / "scaffold_split_results.csv", index=False)
    print(f"\n=== summary over {args.n_repeats} repeats ===")
    print(f"random-split MAE:   mean={out['random_MAE'].mean():.3f} std={out['random_MAE'].std():.3f}")
    print(f"scaffold-split MAE: mean={out['scaffold_MAE'].mean():.3f} std={out['scaffold_MAE'].std():.3f}")
    print(f"generalization gap: mean={out['gap'].mean():+.3f} std={out['gap'].std():.3f}")
    print(f"\nwrote results to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
