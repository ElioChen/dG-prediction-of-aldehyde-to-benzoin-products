"""Molecule-level and donor/acceptor cold-start splits for BDE / cross-benzoin models.

BDE_prediction.md (Phase 1) flags that the project's existing CV (delta_core.cv_evaluate,
train_surrogate.py, ...) is all RepeatedKFold over ROWS -- fine for the homo dG model where
each row is one aldehyde, but for anything donor/acceptor-paired (cross products, and any
BDE model meant to transfer to cross) a random row split leaks: the same donor or acceptor
molecule can appear in both train and test, just paired with a different partner, which
overstates generalization to genuinely new chemistry. This module provides the strict
alternative: assign whole MOLECULES to train/test first, then derive pair membership --
never row-level shuffling for paired data.

Two entry points:
  molecule_cold_split(ids, ...)         -- single-molecule datasets (aldehyde C-H BDE).
  donor_acceptor_cold_split(df, ...)    -- paired datasets (product C-C BDE, cross dG).
      Splits the union of donor/acceptor ids once, then buckets each pair into:
        train          -- both donor and acceptor in the train molecule pool
        test_new_donor     -- donor unseen, acceptor seen in train
        test_new_acceptor  -- acceptor unseen, donor seen in train
        test_both_new      -- neither seen in train (the double cold-start case)
      Matches the eval matrix in BDE_prediction.md section 六: "新 donor；新 acceptor；双新组分".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def molecule_cold_split(ids: pd.Series | np.ndarray, test_frac: float = 0.15,
                         val_frac: float = 0.0, seed: int = 0) -> pd.Series:
    """Assign each *unique* id to train/val/test; return a Series of labels aligned to `ids`
    (duplicates in `ids` get the same label as their first occurrence's molecule)."""
    ids = pd.Series(ids).astype(str)
    uniq = np.sort(ids.unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_test = int(round(len(uniq) * test_frac))
    n_val = int(round(len(uniq) * val_frac))
    label = np.full(len(uniq), "train", dtype=object)
    label[perm[:n_test]] = "test"
    label[perm[n_test:n_test + n_val]] = "val"
    id_to_label = dict(zip(uniq, label))
    return ids.map(id_to_label)


def donor_acceptor_cold_split(df: pd.DataFrame, donor_col: str = "donor_id",
                               acceptor_col: str = "acceptor_id", test_frac: float = 0.2,
                               seed: int = 0) -> pd.Series:
    """Return a Series of {"train","test_new_donor","test_new_acceptor","test_both_new"}
    aligned to `df`'s index. `test_frac` is applied to the UNION molecule pool, not to rows
    -- the resulting row-level test fraction will typically be larger (pairs are quadratic
    in the number of held-out molecules)."""
    donors = df[donor_col].astype(str)
    acceptors = df[acceptor_col].astype(str)
    pool = pd.Index(sorted(set(donors) | set(acceptors)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))
    n_test = int(round(len(pool) * test_frac))
    test_mols = set(pool[perm[:n_test]])

    d_test = donors.isin(test_mols)
    a_test = acceptors.isin(test_mols)
    label = pd.Series("train", index=df.index, dtype=object)
    label[d_test & a_test] = "test_both_new"
    label[d_test & ~a_test] = "test_new_donor"
    label[~d_test & a_test] = "test_new_acceptor"
    return label


def leakage_report(df: pd.DataFrame, split: pd.Series, donor_col: str = "donor_id",
                    acceptor_col: str = "acceptor_id") -> dict:
    """Sanity-check a donor_acceptor_cold_split output: train molecules must be disjoint
    from every test bucket's *held-out* side (the side that made it a test bucket)."""
    train_mols = set(df.loc[split == "train", donor_col].astype(str)) | \
                 set(df.loc[split == "train", acceptor_col].astype(str))
    both_new_mols = set(df.loc[split == "test_both_new", donor_col].astype(str)) | \
                    set(df.loc[split == "test_both_new", acceptor_col].astype(str))
    leaked = train_mols & both_new_mols
    return {
        "counts": split.value_counts().to_dict(),
        "train_molecule_pool": len(train_mols),
        "test_both_new_molecule_pool": len(both_new_mols),
        "leaked_molecules": len(leaked),
        "ok": len(leaked) == 0,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--donor-col", default="donor_id")
    ap.add_argument("--acceptor-col", default="acceptor_id")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    split = donor_acceptor_cold_split(df, args.donor_col, args.acceptor_col,
                                       args.test_frac, args.seed)
    report = leakage_report(df, split, args.donor_col, args.acceptor_col)
    print(report)
    if args.out:
        df = df.copy()
        df["split"] = split.values
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
