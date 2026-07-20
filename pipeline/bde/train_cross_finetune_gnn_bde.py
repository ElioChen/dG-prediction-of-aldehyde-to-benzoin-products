#!/usr/bin/env python
"""B6 (GNN+3D-descriptor hybrid) homo->cross transfer for product BDE (2026-07-16).

The GNN analogue of train_cross_finetune_bde.py (which did this with XGB/H-SPOC and found
C fine-tune > A cross-only > B naive-joint > zero-shot, reproduced at round5 scale --
PROGRESS_20260714.md 〇-3 sec 5). Here the transferred model is B6 -- the current BDE
champion (aldehydes MAE 1.10, products 2.12) -- to see whether the strong hybrid model
transfers homo->cross better than the XGB version did.

Feature-dimension constraint: the cross training table only carries the 36 descriptors shared
with the homo product side (no ADCH/QTAIM -- see DESCRIPTOR_POLICY_CROSS.md). So B6 is
pretrained here on homo products using ONLY those 36 shared features (a restricted-x_d B6,
not the full-43-feature champion) so the graph encoder + x_d head have a matching input
dimension on both homo and cross. This makes the homo pretrain here slightly weaker than the
headline B6, but it's the only apples-to-apples way to warm-start onto cross.

Variants, all scored on the SAME held-out cross test split (donor/acceptor double-cold):
  zero_shot : homo-pretrained B6, no cross training, predict cross test.
  A_cross_only : fresh B6 trained on cross train only.
  C_finetune   : homo-pretrained B6 weights, warm-started (continue trainer.fit) on cross train.
(B naive-joint is intentionally skipped: it was the worst variant for XGB and the dilution
mechanism -- 176k homo swamping ~10k cross -- is architecture-independent, so paying a second
~4h homo-scale GNN training for a near-certain "worst" result isn't worth the GPU. Noted, not
re-derived.)

The expensive homo pretrain is checkpointed to --pretrain-ckpt and reused if present, so
re-runs (e.g. different cross tables / seeds) are cheap.

Usage:
  python train_cross_finetune_gnn_bde.py \
      --cross-table .../cross_round5/cross_train_table_5rounds_mordred.parquet \
      --pretrain-ckpt .../runs/logs/b6_homo_products_shared36.pt \
      --out .../runs/logs/cross_finetune_gnn_round5.json
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
from splits import molecule_cold_split, donor_acceptor_cold_split  # noqa: E402
from train_gnn_hybrid_bde import (  # noqa: E402
    GLOBAL_XTB, DEFAULT_PARAMS, build_mpnn, make_dataset, train_one, predict,
)

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

# The 36 product-side descriptors present, unprefixed, on BOTH the homo products table and
# the cross training table (identical set to train_cross_finetune_bde.SHARED_FEATURES).
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


def load_cross(cross_table, seed, test_frac):
    usecols = ["donor_id", "acceptor_id", "smiles", "bde_gxtb_kcal"] + SHARED_FEATURES
    if str(cross_table).endswith(".parquet"):
        df = pd.read_parquet(cross_table)[usecols].copy()
        df["donor_id"] = df["donor_id"].astype(str)
        df["acceptor_id"] = df["acceptor_id"].astype(str)
    else:
        df = pd.read_csv(cross_table, usecols=usecols,
                          dtype={"donor_id": str, "acceptor_id": str}, low_memory=False)
    for c in SHARED_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bde_gxtb_kcal"])
    df = df[qc_filter(df["bde_gxtb_kcal"])]
    df = df[df["smiles"].astype(str) != ""]
    df = df.dropna(subset=SHARED_FEATURES, how="all").reset_index(drop=True)
    split = donor_acceptor_cold_split(df, test_frac=test_frac, seed=seed)
    return df, split


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


def build_dsets(smiles, Xs, ysc, tr_in, val_idx, eval_idx, featurizer):
    tr = make_dataset(smiles[tr_in], Xs[tr_in], ysc[tr_in], featurizer)
    val = make_dataset(smiles[val_idx], Xs[val_idx], ysc[val_idx], featurizer)
    ev = make_dataset(smiles[eval_idx], Xs[eval_idx], None, featurizer)
    return tr, val, ev


def main():
    import torch
    from chemprop import featurizers

    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-table", type=Path, required=True)
    ap.add_argument("--pretrain-ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--homo-test-frac", type=float, default=0.15)
    ap.add_argument("--cross-test-frac", type=float, default=0.2)
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

    # ---- cross side ----
    cross_df, cross_split = load_cross(args.cross_table, args.seed, args.cross_test_frac)
    cross_tr_mask = (cross_split == "train").to_numpy()
    cross_te_idx = np.where(~cross_tr_mask)[0]
    cross_tr_idx = np.where(cross_tr_mask)[0]
    c_tr_in, c_val = carve_val(cross_tr_idx, args.seed, params["val_frac"])
    Xc, yc, smi_c = prep_xy(cross_df, SHARED_FEATURES)
    print(f"cross: n={len(cross_df)} train={cross_tr_mask.sum()} test={len(cross_te_idx)} "
          f"({cross_split[~cross_tr_mask].value_counts().to_dict()})", flush=True)

    results = {"n_homo_train": int(len(h_tr_in)), "n_cross_train": int(cross_tr_mask.sum()),
               "n_cross_test": int(len(cross_te_idx)),
               "cross_test_bucket_counts": cross_split[~cross_tr_mask].value_counts().to_dict(),
               "shared_features": len(SHARED_FEATURES)}

    from lightning import pytorch as pl

    def predict_with(model, X_scaled_by, y_scaler, idx):
        """Predict on cross rows `idx`, scaling x_d with `X_scaled_by` scaler and
        inverse-transforming with `y_scaler`."""
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

    # C finetune: warm-start homo weights, continue on cross train (homo scalers kept, so the
    # pretrained x_d head sees consistently-scaled inputs; fine-tuning adapts the rest).
    ft_model = build_mpnn(n_xd, params)
    ft_model.load_state_dict(pre_model.state_dict())
    ft_params = dict(params, max_epochs=args.finetune_epochs)
    tr_d = make_dataset(smi_c[c_tr_in], xs.transform(Xc[c_tr_in]),
                        ys.transform(yc[c_tr_in].reshape(-1, 1)).ravel(), featurizer)
    val_d = make_dataset(smi_c[c_val], xs.transform(Xc[c_val]),
                         ys.transform(yc[c_val].reshape(-1, 1)).ravel(), featurizer)
    ft_model, ft_trainer = train_one(tr_d, val_d, ft_params, args.seed, n_xd,
                                     model_override=ft_model)
    results["C_homo_pretrain_cross_finetune"] = evaluate(
        yc_te, predict_with(ft_model, xs, ys, cross_te_idx), yc[c_tr_in].mean())
    print("C_finetune:", results["C_homo_pretrain_cross_finetune"], flush=True)

    # A cross-only: fresh B6 on cross train, cross-fit scalers
    xs_c = StandardScaler().fit(Xc[c_tr_in])
    ys_c = StandardScaler().fit(yc[c_tr_in].reshape(-1, 1))
    tr_d = make_dataset(smi_c[c_tr_in], xs_c.transform(Xc[c_tr_in]),
                        ys_c.transform(yc[c_tr_in].reshape(-1, 1)).ravel(), featurizer)
    val_d = make_dataset(smi_c[c_val], xs_c.transform(Xc[c_val]),
                         ys_c.transform(yc[c_val].reshape(-1, 1)).ravel(), featurizer)
    a_model, _ = train_one(tr_d, val_d, dict(params, max_epochs=args.finetune_epochs),
                           args.seed, n_xd)
    results["A_cross_only"] = evaluate(
        yc_te, predict_with(a_model, xs_c, ys_c, cross_te_idx), yc[c_tr_in].mean())
    print("A_cross_only:", results["A_cross_only"], flush=True)

    print(json.dumps(results, indent=2))
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
