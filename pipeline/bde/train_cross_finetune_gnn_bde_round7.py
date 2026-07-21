#!/usr/bin/env python
"""B6 (GNN+3D-descriptor hybrid) homo->cross transfer for product BDE, round1-7 scale
(2026-07-17).

Direct follow-up to train_cross_finetune_gnn_bde.py (which ran this at round5 scale, n=15820
QC'd rows, donor/acceptor-cold split built on the fly -- PROGRESS_20260714.md O-4 "B6 GNN版
交叉微调"). That experiment was explicitly flagged as blocked on "等隔壁round6/round7交叉表
就绪" -- cross-benzoin now has round1-7 (32,456 raw rows), 3.2x more than round5's 10,107
train rows, so this reruns the same zero-shot / A cross-only / C fine-tune comparison at
that scale.

Two differences from the round5 script, both deliberate:

1. **Split source**: round5 used `donor_acceptor_cold_split()` computed fresh inside the
   script. Since then cross-benzoin discovered its OWN molecule-cold split leaked scaffolds
   (83-93% of "held-out" molecules shared a Murcko scaffold with train -- see memory
   [[cross-five-diagnostics-20260717]] / [[cross-scaffold-disjoint-rebuild-20260717]]) and
   rebuilt a genuinely scaffold-disjoint split, re-labeling round1-7's full table with a
   `new_scaffold_split` column (train=19,687 / validation=483 / test=450 / mixed=11,836
   discarded -- a pair is "mixed" if its donor and acceptor scaffolds land in different
   buckets). This script consumes that column directly (via --split-col) instead of deriving
   its own split, since it's now the more trustworthy split for cross-benzoin's own data and
   all needed columns (id, smiles, bde_gxtb_kcal, the 36 shared descriptors, split label) are
   present -- no detour needed. This trades away exact backward-comparability of absolute
   MAE against the round3/round5 numbers (different, incomparable test sets, as those two were
   already flagged as non-comparable to each other) in exchange for both a much larger test
   set (450 vs round5's 5713 -- actually smaller in row count, but the OLD round5 split was
   itself leaky/inflated, unverified) and a leak-free evaluation. Rank ordering (C vs A vs
   zero-shot) is what's being tracked across rounds, not the absolute number.
2. **B naive-merge intentionally still skipped**, same reasoning as the round5 GNN script:
   the XGB version already showed B < A robustly at two data scales (round3, round5) and the
   dilution mechanism (~176k homo rows swamping ~18-20k cross rows) is architecture-
   independent, so a second ~4h homo-scale GNN pretrain paying for a near-certain "worst"
   result isn't worth the GPU here either.

The cross CSV is a pre-materialized, column-pruned copy of cross-benzoin's own
`cross_train_table_7rounds_scaffold_split_labeled.parquet` (envs/bde_gnn has no pyarrow) --
see `runs/logs/bde_cross_round7_scaffold_split_labeled.csv`, written once by this project,
read-only input to cross-benzoin's own table, does not touch or modify anything in
cross_benzoin's own pipeline/output tree.

Usage:
  python train_cross_finetune_gnn_bde_round7.py \
      --cross-table .../runs/logs/bde_cross_round7_scaffold_split_labeled.csv \
      --pretrain-ckpt .../runs/logs/b6_homo_products_shared36.pt \
      --out .../runs/logs/cross_finetune_gnn_round7_scaffold.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc import qc_filter  # noqa: E402
from splits import molecule_cold_split  # noqa: E402
from train_gnn_hybrid_bde import (  # noqa: E402
    GLOBAL_XTB, DEFAULT_PARAMS, build_mpnn, make_dataset, train_one, predict,
)

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

# Identical to train_cross_finetune_gnn_bde.SHARED_FEATURES / train_cross_finetune_bde's
# 36 shared descriptors -- kept in sync by construction (both scripts import GLOBAL_XTB from
# the same source and hand-list the same local product descriptors below).
SHARED_FEATURES = GLOBAL_XTB + [
    "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO", "mulliken_hydH",
    "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb", "fukui_plus_ketC", "fukui_minus_ketC",
    "fukui_0_ketC", "dual_ketC", "fukui_plus_carbC", "fukui_minus_carbC", "fukui_0_carbC",
    "dual_carbC", "pa_ketO", "vbur_ketC", "vbur_carbC", "sterimol_L", "sterimol_B1",
    "sterimol_B5", "SASA_total", "P_int", "hb_dist", "hb_angle", "dih_core",
]
assert len(SHARED_FEATURES) == 36, len(SHARED_FEATURES)


def load_homo(seed, test_frac):
    labels = pd.read_csv(H / "products_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    labels = labels.dropna(subset=["bde_gxtb_kcal"]).drop_duplicates("id")
    labels = labels[qc_filter(labels["bde_gxtb_kcal"])]
    mol = pd.read_csv(H / "products_all.csv",
                      usecols=["id", "donor_id", "smiles", "error"] + SHARED_FEATURES,
                      dtype=str, keep_default_na=False, low_memory=False)
    mol = mol[(mol["error"] == "") & (mol["smiles"] != "")]
    for c in SHARED_FEATURES:
        mol[c] = pd.to_numeric(mol[c], errors="coerce")
    df = labels.merge(mol, on="id", how="inner").dropna(subset=SHARED_FEATURES, how="all")
    df = df.reset_index(drop=True)
    split = molecule_cold_split(df["donor_id"], test_frac=test_frac, seed=seed)
    return df, (split == "train").to_numpy()


def load_cross_scaffold(cross_table, split_col):
    """Load the cross round1-7 table using its PRE-LABELED scaffold-disjoint split column
    (produced by cross-benzoin's own rebuild_scaffold_disjoint_split.py pair-labeling pass)
    instead of deriving a fresh donor/acceptor-cold split. 'mixed' rows (donor and acceptor
    scaffolds land in different buckets) are dropped -- neither safely trainable nor safely
    testable, same logic cross-benzoin itself applied."""
    usecols = ["donor_id", "acceptor_id", "smiles", "bde_gxtb_kcal", split_col] + SHARED_FEATURES
    df = pd.read_csv(cross_table, usecols=usecols,
                      dtype={"donor_id": str, "acceptor_id": str}, low_memory=False)
    for c in SHARED_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bde_gxtb_kcal"])
    df = df[qc_filter(df["bde_gxtb_kcal"])]
    df = df[df["smiles"].astype(str) != ""]
    df = df[df[split_col].isin(["train", "validation", "test"])]  # drop "mixed"
    df = df.dropna(subset=SHARED_FEATURES, how="all").reset_index(drop=True)
    return df


def prep_xy(df, feats, ycol="bde_gxtb_kcal"):
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    return X, y, df["smiles"].to_numpy()


def carve_val(tr_idx, seed, val_frac=0.1):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr_idx)
    n_val = max(1, int(round(val_frac * len(perm))))
    return perm[n_val:], perm[:n_val]


def evaluate(y_true, y_pred, y_train_mean):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "spearman_rho": float(spearmanr(y_true, y_pred).correlation),
        "mean_baseline_MAE": float(mean_absolute_error(
            y_true, np.full(len(y_true), y_train_mean))),
    }


def main():
    import torch
    from chemprop import featurizers

    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-table", type=Path, required=True)
    ap.add_argument("--split-col", default="new_scaffold_split")
    ap.add_argument("--pretrain-ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--homo-test-frac", type=float, default=0.15)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subsample + 2 epochs to validate the code path end-to-end")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = dict(DEFAULT_PARAMS)
    if args.smoke:
        params["max_epochs"] = 2
        args.finetune_epochs = 2
    n_xd = len(SHARED_FEATURES)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # ---- homo side: fit scalers on homo train, pretrain (or load ckpt) ----
    homo_df, homo_tr_mask = load_homo(args.seed, args.homo_test_frac)
    if args.smoke:
        keep = np.zeros(len(homo_df), dtype=bool)
        keep[np.random.default_rng(0).choice(len(homo_df), 800, replace=False)] = True
        homo_df = homo_df[keep].reset_index(drop=True)
        homo_tr_mask = homo_tr_mask[keep]
    Xh, yh, smi_h = prep_xy(homo_df, SHARED_FEATURES)
    homo_tr_idx = np.where(homo_tr_mask)[0]
    h_tr_in, h_val = carve_val(homo_tr_idx, args.seed, params["val_frac"])
    xs = StandardScaler().fit(Xh[h_tr_in])
    ys = StandardScaler().fit(yh[h_tr_in].reshape(-1, 1))
    Xh_s = xs.transform(Xh)
    yh_s = ys.transform(yh.reshape(-1, 1)).ravel()
    print(f"homo products: n={len(homo_df)} train={len(h_tr_in)} val={len(h_val)}", flush=True)

    pre_model = build_mpnn(n_xd, params)
    if args.pretrain_ckpt.exists():
        pre_model.load_state_dict(torch.load(args.pretrain_ckpt, map_location="cpu"))
        print(f"loaded pretrained B6 from {args.pretrain_ckpt}", flush=True)
    else:
        tr_d = make_dataset(smi_h[h_tr_in], Xh_s[h_tr_in], yh_s[h_tr_in], featurizer)
        val_d = make_dataset(smi_h[h_val], Xh_s[h_val], yh_s[h_val], featurizer)
        pre_model, _ = train_one(tr_d, val_d, params, args.seed, n_xd)
        args.pretrain_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pre_model.state_dict(), args.pretrain_ckpt)
        print(f"pretrained + saved B6 to {args.pretrain_ckpt}", flush=True)

    # ---- cross side: pre-labeled scaffold-disjoint split ----
    cross_df = load_cross_scaffold(args.cross_table, args.split_col)
    if args.smoke:
        keep = np.zeros(len(cross_df), dtype=bool)
        te_idx = np.where(cross_df[args.split_col].to_numpy() == "test")[0]
        tr_idx = np.where(cross_df[args.split_col].to_numpy() == "train")[0]
        val_idx = np.where(cross_df[args.split_col].to_numpy() == "validation")[0]
        rng = np.random.default_rng(0)
        keep[rng.choice(tr_idx, min(600, len(tr_idx)), replace=False)] = True
        keep[val_idx] = True
        keep[te_idx] = True
        cross_df = cross_df[keep].reset_index(drop=True)
    split_labels = cross_df[args.split_col].to_numpy()
    cross_tr_idx = np.where(split_labels == "train")[0]
    cross_val_idx = np.where(split_labels == "validation")[0]
    cross_te_idx = np.where(split_labels == "test")[0]
    Xc, yc, smi_c = prep_xy(cross_df, SHARED_FEATURES)
    print(f"cross (scaffold-disjoint): n={len(cross_df)} train={len(cross_tr_idx)} "
          f"validation={len(cross_val_idx)} test={len(cross_te_idx)}", flush=True)

    results = {"n_homo_train": int(len(h_tr_in)), "n_cross_train": int(len(cross_tr_idx)),
               "n_cross_validation": int(len(cross_val_idx)),
               "n_cross_test": int(len(cross_te_idx)),
               "split_col": args.split_col, "shared_features": len(SHARED_FEATURES)}

    from lightning import pytorch as pl

    def predict_with(model, X_scaled_by, y_scaler, idx):
        ev = make_dataset(smi_c[idx], X_scaled_by.transform(Xc[idx]), None, featurizer)
        loader_trainer = pl.Trainer(accelerator="auto", devices=1, logger=False,
                                    enable_progress_bar=False, enable_checkpointing=False)
        from chemprop import data
        loader = data.build_dataloader(ev, batch_size=params["batch_size"],
                                        shuffle=False, num_workers=0)
        preds = loader_trainer.predict(model, loader)
        p = np.concatenate([q.numpy() for q in preds]).ravel()
        return y_scaler.inverse_transform(p.reshape(-1, 1)).ravel()

    yc_te = yc[cross_te_idx]

    # zero-shot: homo model + homo scalers, predict cross test
    results["zero_shot_homo_only"] = evaluate(
        yc_te, predict_with(pre_model, xs, ys, cross_te_idx), yh[h_tr_in].mean())
    print("zero_shot:", results["zero_shot_homo_only"], flush=True)

    # C finetune: warm-start homo weights, continue on cross train, early-stop on the
    # PROVIDED validation bucket (not a re-carved slice of train -- the split already
    # supplies one, use it directly).
    ft_model = build_mpnn(n_xd, params)
    ft_model.load_state_dict(pre_model.state_dict())
    ft_params = dict(params, max_epochs=args.finetune_epochs)
    tr_d = make_dataset(smi_c[cross_tr_idx], xs.transform(Xc[cross_tr_idx]),
                        ys.transform(yc[cross_tr_idx].reshape(-1, 1)).ravel(), featurizer)
    val_d = make_dataset(smi_c[cross_val_idx], xs.transform(Xc[cross_val_idx]),
                         ys.transform(yc[cross_val_idx].reshape(-1, 1)).ravel(), featurizer)
    ft_model, ft_trainer = train_one(tr_d, val_d, ft_params, args.seed, n_xd,
                                     model_override=ft_model)
    results["C_homo_pretrain_cross_finetune"] = evaluate(
        yc_te, predict_with(ft_model, xs, ys, cross_te_idx), yc[cross_tr_idx].mean())
    print("C_finetune:", results["C_homo_pretrain_cross_finetune"], flush=True)

    # A cross-only: fresh B6 on cross train, cross-fit scalers, same provided validation bucket
    xs_c = StandardScaler().fit(Xc[cross_tr_idx])
    ys_c = StandardScaler().fit(yc[cross_tr_idx].reshape(-1, 1))
    tr_d = make_dataset(smi_c[cross_tr_idx], xs_c.transform(Xc[cross_tr_idx]),
                        ys_c.transform(yc[cross_tr_idx].reshape(-1, 1)).ravel(), featurizer)
    val_d = make_dataset(smi_c[cross_val_idx], xs_c.transform(Xc[cross_val_idx]),
                         ys_c.transform(yc[cross_val_idx].reshape(-1, 1)).ravel(), featurizer)
    a_model, _ = train_one(tr_d, val_d, dict(params, max_epochs=args.finetune_epochs),
                           args.seed, n_xd)
    results["A_cross_only"] = evaluate(
        yc_te, predict_with(a_model, xs_c, ys_c, cross_te_idx), yc[cross_tr_idx].mean())
    print("A_cross_only:", results["A_cross_only"], flush=True)

    print(json.dumps(results, indent=2))
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
