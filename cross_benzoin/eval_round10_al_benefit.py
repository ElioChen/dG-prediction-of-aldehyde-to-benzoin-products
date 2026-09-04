#!/usr/bin/env python
"""Does training on the round10 active-learning batch actually improve the model
on round10-like (AL-hard) pairs?

The frozen n=448 scaffold-disjoint holdout can't answer this -- it has no
round10-like scaffolds. So: hold out a random 30% of the round10 labelled rows as
a fixed "AL-hard test", train the MLP+XGB ensemble twice (with / without the other
70% of round10), and compare MAE on that held-out slice.

  Model A  = ensemble on (r1-9 train rows only)          [no round10]
  Model B  = ensemble on (r1-9 train rows + 70% round10) [round10 AL batch]
  test     = held-out 30% round10   AND   frozen n=448

If MAE_B < MAE_A on the round10 test -> ingesting the AL batch generalises to
similar hard pairs -> AL delivers, not just diagnoses.

Ensemble-only (no GNN retrain per arm -- too slow); the champion's blend adds a
roughly constant ~0.05-0.1 on top, so the A-vs-B delta is the signal.

  /home/schen3/venv/nequip/bin/python cross_benzoin/eval_round10_al_benefit.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "cross_benzoin"))
import delta_core as dc
from train_cross_delta import _feature_blocks, TARGET_COL, BASELINE_COL
from train_cross_ensemble import _fit_ensemble

TABLE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
SEED = 42
HOLDOUT_FRAC = 0.30


def _fit_predict(train_df, tests, feats, seed):
    med = train_df[feats].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    Xtr = train_df[feats].apply(pd.to_numeric, errors="coerce").fillna(med)
    ytr = (train_df[TARGET_COL] - train_df[BASELINE_COL]).to_numpy()
    sc, mlp, xa, xb = _fit_ensemble(Xtr, ytr, seed)
    out = {}
    for name, te in tests.items():
        Xte = te[feats].apply(pd.to_numeric, errors="coerce").fillna(med)
        dd = (mlp.predict(sc.transform(Xte)) + xa.predict(Xte) + xb.predict(Xte)) / 3.0
        pred = te[BASELINE_COL].to_numpy() + dd
        out[name] = dc.metrics_vs_dft(te[TARGET_COL].to_numpy(), pred)
    return out


def main() -> int:
    df = pd.read_parquet(TABLE)
    rnd = df["round"].astype(str)
    is_r10 = rnd.str.contains("10")
    trainable = df["new_scaffold_split"] == "train"
    frozen_test = df[df["new_scaffold_split"] == "test"].reset_index(drop=True)

    r10_train = df[is_r10 & trainable].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    held = rng.random(len(r10_train)) < HOLDOUT_FRAC
    r10_test = r10_train[held].reset_index(drop=True)
    r10_keep = r10_train[~held].reset_index(drop=True)
    base = df[(~is_r10) & trainable].reset_index(drop=True)
    print(f"non-round10 trainable: {len(base)}  |  round10 trainable: {len(r10_train)} "
          f"-> keep {len(r10_keep)} / hard-test {len(r10_test)}  |  frozen n={len(frozen_test)}")

    all_feats = [c for c in df.columns if c not in {
        "id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
        "donor_smiles", "acceptor_smiles", "smiles", "dG_xtb_kcal", "dG_gxtb_kcal",
        "dG_orca_kcal", "donor_scaf_split", "acceptor_scaf_split", "new_scaffold_split"}]
    feats = _feature_blocks(all_feats)["all_raw_blocks+mordred"]

    tests = {"r10_hard_test": r10_test, "frozen_448": frozen_test}
    gx = {k: dc.metrics_vs_dft(v[TARGET_COL].to_numpy(), v[BASELINE_COL].to_numpy())["MAE"]
          for k, v in tests.items()}

    print("\n--- Model A: ensemble WITHOUT round10 ---")
    A = _fit_predict(base, tests, feats, SEED)
    print("\n--- Model B: ensemble WITH 70% round10 ---")
    B = _fit_predict(pd.concat([base, r10_keep], ignore_index=True), tests, feats, SEED)

    print("\n================ RESULT ================")
    print(f"{'test':16s} {'g-xTB':>8s} {'A (no r10)':>12s} {'B (+r10)':>12s} {'B-A':>8s}")
    for k in tests:
        d = B[k]["MAE"] - A[k]["MAE"]
        print(f"{k:16s} {gx[k]:8.3f} {A[k]['MAE']:12.3f} {B[k]['MAE']:12.3f} {d:+8.3f}")
    print("\nB-A < 0 on r10_hard_test  => training on the AL batch helps on AL-hard pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
