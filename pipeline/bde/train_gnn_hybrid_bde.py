#!/usr/bin/env python
"""Phase-1 baseline B6 (2026-07-15, user-requested GNN-architecture exploration): hybrid
D-MPNN + local-3D-descriptor fusion for BDE, direct analogue of the main dG model's
GNN+tabular hybrid (`pipeline/gnn/gnn_core.py`) which found a real (if small) MAE gain from
stacking -- that project used Chemprop's `x_d` mechanism to concatenate 62 hand-built
descriptors onto the graph embedding before the FFN readout, rather than treating "GNN vs
tree model" as an either/or choice.

B4 (train_gnn_bde.py) is graph-ONLY: no atom/bond features beyond chemprop's default 2D
featurizer, so it cannot see the local electronic-structure signal at the reacting atom that
H-SPOC (train_local3d_baseline.py: Fukui indices, WBO, ADCH/QTAIM charges, vbur, sterimol --
tuned XGB now hits R^2 0.76/0.82) demonstrably carries. This fuses the two: same
BondMessagePassing+MeanAggregation graph encoder as B4, same LOCAL_FEATURES descriptor set
as H-SPOC, concatenated via `x_d` exactly like gnn_core.py does for the production dG model
-- no new compute, both halves already exist on disk.

Motivation for trying this specifically (vs. e.g. attention pooling or a 3D-coordinate
model): it's the one architecture change in this project with an ALREADY-PROVEN track
record (gnn_core.py's stacking gain on the production model) rather than a first attempt,
and the ingredients are free. A 3D-coordinate-native model (SchNet/DimeNet-style) remains a
bigger, separate investment -- noted as a follow-up, not attempted here.

Usage:
  python train_gnn_hybrid_bde.py --which aldehydes --out /tmp/b6_ald.json
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
from qc import qc_filter
from splits import molecule_cold_split

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

# Duplicated from train_local3d_baseline.py (H-SPOC), not imported: that module pulls in
# xgboost at load time, which isn't installed in this script's env (envs/bde_gnn is
# torch/chemprop-only, matching the project's per-baseline isolated-env convention -- see
# `[[envs-gnn-corrupted]]` memory for why cross-env imports are avoided here). Keep in sync
# with LOCAL_FEATURES in train_local3d_baseline.py if either changes.
GLOBAL_XTB = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta",
              "xtb_omega", "xtb_dipole"]
LOCAL_FEATURES = {
    "aldehydes": GLOBAL_XTB + [
        "mulliken_CHO_C", "mulliken_CHO_O", "fukui_plus_CHO_C", "fukui_minus_CHO_C",
        "fukui_0_CHO_C", "dual_descriptor_CHO_C", "wbo_CO", "pa_CHO_O", "vbur_CHO_C",
        "sterimol_L", "sterimol_B1", "sterimol_B5", "SASA_total", "P_int",
        "adch_CHO_C", "adch_CHO_O", "adch_fukui_plus_CHO_C", "adch_fukui_minus_CHO_C",
        "adch_fukui_minus_CHO_O", "qtaim_lap_CO", "qtaim_ell_CO",
    ],
    "products": GLOBAL_XTB + [
        "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO", "mulliken_hydH",
        "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb", "fukui_plus_ketC", "fukui_minus_ketC",
        "fukui_0_ketC", "dual_ketC", "fukui_plus_carbC", "fukui_minus_carbC", "fukui_0_carbC",
        "dual_carbC", "pa_ketO", "vbur_ketC", "vbur_carbC", "sterimol_L", "sterimol_B1",
        "sterimol_B5", "SASA_total", "P_int", "hb_dist", "hb_angle", "dih_core",
        "adch_ketC", "adch_ketO", "adch_carbC", "adch_hydO", "adch_hydH",
        "adch_fukui_plus_ketC", "adch_fukui_minus_ketC", "adch_fukui_plus_carbC",
        "adch_fukui_minus_carbC", "qtaim_rho_CO_ket", "qtaim_lap_CO_ket", "qtaim_ell_CO_ket",
        "qtaim_rho_CC_new", "qtaim_lap_CC_new", "qtaim_ell_CC_new", "qtaim_rho_HB",
    ],
}

DEFAULT_PARAMS = dict(depth=4, message_hidden=300, ffn_hidden=300, ffn_layers=2,
                       dropout=0.0, batch_size=64, max_epochs=120, patience=20,
                       val_frac=0.1)


MP_CLASSES = {"bond": "BondMessagePassing", "atom": "AtomMessagePassing",
              "mab_bond": "MABBondMessagePassing", "mab_atom": "MABAtomMessagePassing"}
AGG_CLASSES = {"mean": "MeanAggregation", "sum": "SumAggregation",
               "norm": "NormAggregation", "attentive": "AttentiveAggregation"}


def build_mpnn(n_xd: int, params):
    from chemprop import models, nn as cpnn
    mp_cls = getattr(cpnn, MP_CLASSES[params.get("message_passing", "bond")])
    mp = mp_cls(depth=params["depth"], d_h=params["message_hidden"])
    # MAB* variants report (vertex_dim, edge_dim) via output_dims instead of a single
    # output_dim (they can return both vertex and edge embeddings); pooling uses the
    # vertex/atom embedding, i.e. output_dims[0].
    mp_out_dim = mp.output_dim if hasattr(mp, "output_dim") else mp.output_dims[0]
    agg_name = params.get("aggregation", "mean")
    if agg_name == "attentive":
        agg = cpnn.AttentiveAggregation(output_size=mp_out_dim)
    else:
        agg = getattr(cpnn, AGG_CLASSES[agg_name])()
    ffn = cpnn.RegressionFFN(input_dim=mp_out_dim + n_xd, hidden_dim=params["ffn_hidden"],
                              n_layers=params["ffn_layers"], dropout=params["dropout"])
    return models.MPNN(mp, agg, ffn, batch_norm=True)


def make_dataset(smiles, X, y, featurizer):
    from chemprop import data
    dps = []
    for i, smi in enumerate(smiles):
        yi = None if y is None else np.array([y[i]], dtype=float)
        dps.append(data.MoleculeDatapoint.from_smi(smi, y=yi, x_d=np.asarray(X[i], dtype=float)))
    return data.MoleculeDataset(dps, featurizer)


def train_one(tr_dset, val_dset, params, seed, n_xd, model_override=None):
    from chemprop import data
    from lightning import pytorch as pl
    pl.seed_everything(seed, workers=True)
    # model_override: pass a pre-built (e.g. warm-started/fine-tune) MPNN to continue
    # training instead of a fresh random init -- used by train_cross_finetune_gnn_bde.py.
    model = model_override if model_override is not None else build_mpnn(n_xd, params)
    tr_loader = data.build_dataloader(tr_dset, batch_size=params["batch_size"],
                                       shuffle=True, num_workers=0)
    val_loader = data.build_dataloader(val_dset, batch_size=params["batch_size"],
                                        shuffle=False, num_workers=0)
    cbs = [pl.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                       patience=params["patience"])]
    trainer = pl.Trainer(max_epochs=params["max_epochs"], accelerator="auto", devices=1,
                          enable_progress_bar=False, enable_checkpointing=False,
                          logger=False, callbacks=cbs, deterministic=False)
    trainer.fit(model, tr_loader, val_loader)
    return model, trainer


def predict(model, trainer, dset, params):
    from chemprop import data
    loader = data.build_dataloader(dset, batch_size=params["batch_size"],
                                    shuffle=False, num_workers=0)
    preds = trainer.predict(model, loader)
    return np.concatenate([p.numpy() for p in preds]).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--target", choices=["bde", "bdfe"], default="bde")
    ap.add_argument("--n", type=int, default=None, help="subsample for smoke tests")
    ap.add_argument("--train-frac", type=float, default=None,
                     help="subsample ONLY the train split to this fraction (test/val stay "
                          "full-size and identical across fractions) -- for learning-curve "
                          "experiments, unlike --n which shrinks train+test together")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-out", default=None)
    ap.add_argument("--save-checkpoint", default=None,
                     help="if set, save trained model state_dict + x/y scalers + feature "
                          "list + hyperparams + split-file provenance to this .pt path "
                          "(default: off, existing behavior unchanged) -- 2026-07-17, added "
                          "to package a deployable champion checkpoint (see "
                          "predict_bde_champion.py); enable_checkpointing stays False, this "
                          "is a manual one-shot save of the final trained model, not "
                          "mid-training checkpointing")
    # hyperparameter overrides (for tuning search); default = DEFAULT_PARAMS
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--message-hidden", type=int, default=None)
    ap.add_argument("--ffn-hidden", type=int, default=None)
    ap.add_argument("--ffn-layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--message-passing", choices=list(MP_CLASSES), default="bond",
                     help="bond/atom = chemprop's standard MPNN variants (B4/B6 default is "
                          "bond); mab_bond/mab_atom = attention-based message passing "
                          "(chemprop>=2.3's MAB* classes) -- a genuinely different GNN "
                          "architecture, not just a hyperparameter tweak")
    ap.add_argument("--aggregation", choices=list(AGG_CLASSES), default="mean",
                     help="mean = B6 default; attentive = learned attention-weighted pooling "
                          "instead of a fixed mean/sum over atoms")
    # --eval-on val reports on the internal validation fold (carved from train) and never
    # touches the test set -- use this for hyperparameter selection so test stays frozen.
    # The chosen config is then re-run once with --eval-on test for the honest number.
    ap.add_argument("--eval-on", choices=["test", "val"], default="test")
    ap.add_argument("--split-file", type=Path, default=None,
                     help="if set, use this precomputed id->scaffold_split parquet "
                          "(columns: id, scaffold_split in {train,validation,test} or "
                          "{train,test}) INSTEAD of molecule_cold_split -- for a genuinely "
                          "scaffold-disjoint split (2026-07-17), not just molecule-disjoint. "
                          "'validation' rows (if present) are folded into the train pool "
                          "(this script carves its own early-stopping val internally via "
                          "params['val_frac']); only 'test' is held out.")
    args = ap.parse_args()
    params = dict(DEFAULT_PARAMS)
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
    for k, v in [("depth", args.depth), ("message_hidden", args.message_hidden),
                 ("ffn_hidden", args.ffn_hidden), ("ffn_layers", args.ffn_layers),
                 ("dropout", args.dropout), ("batch_size", args.batch_size)]:
        if v is not None:
            params[k] = v
    params["message_passing"] = args.message_passing
    params["aggregation"] = args.aggregation

    feats = LOCAL_FEATURES[args.which]
    labels = pd.read_csv(H / f"{args.which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    ycol = f"{args.target}_gxtb_kcal"
    labels = labels.dropna(subset=[ycol]).drop_duplicates("id")
    labels = labels[qc_filter(labels[ycol])]

    id_cols = ["id", "smiles", "error"] + feats if args.which == "aldehydes" \
        else ["id", "donor_id", "smiles", "error"] + feats
    mol = pd.read_csv(H / f"{args.which}_all.csv", usecols=id_cols, dtype=str,
                       keep_default_na=False, low_memory=False)
    mol = mol[mol["error"] == ""]
    for c in feats:
        mol[c] = pd.to_numeric(mol[c], errors="coerce")

    df = labels.merge(mol, on="id", how="inner").dropna(subset=feats, how="all")
    df = df[df["smiles"] != ""].reset_index(drop=True)
    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"{args.which}: {len(df)} rows with BDE label + smiles + local-3D descriptors "
          f"({len(feats)} x_d features)", flush=True)

    split_col = "id" if args.which == "aldehydes" else "donor_id"
    if args.split_file is not None:
        # split file has one row per THIS row's own "id" (already correctly grouped by
        # parent/donor scaffold at build time, see build_scaffold_splits.py) -- merge on
        # "id" directly, not split_col (which is only relevant to molecule_cold_split's
        # own internal grouping).
        read = pd.read_csv if args.split_file.suffix == ".csv" else pd.read_parquet
        sf = read(args.split_file)[["id", "scaffold_split"]]
        sf["id"] = sf["id"].astype(str)
        df = df.merge(sf, on="id", how="left")
        n_unmatched = df["scaffold_split"].isna().sum()
        if n_unmatched:
            print(f"WARN: {n_unmatched}/{len(df)} rows have no scaffold_split assignment "
                  f"(dropped)", flush=True)
        df = df.dropna(subset=["scaffold_split"]).reset_index(drop=True)
        split = df["scaffold_split"].replace({"validation": "train"})
        print(f"{args.which}: using scaffold-disjoint --split-file {args.split_file.name}", flush=True)
    else:
        split = molecule_cold_split(df[split_col], test_frac=args.test_frac, seed=args.seed)
    tr_mask, te_mask = (split == "train").to_numpy(), (split == "test").to_numpy()
    print(f"{args.which}: n={len(df)}  train={tr_mask.sum()}  test={te_mask.sum()}  "
          f"cold on '{split_col}'", flush=True)

    rng = np.random.default_rng(args.seed)
    tr_idx = np.where(tr_mask)[0]
    perm = rng.permutation(tr_idx)
    n_val = max(1, int(round(params["val_frac"] * len(perm))))
    val_idx, tr_in = perm[:n_val], perm[n_val:]
    if args.train_frac is not None:
        n_keep = max(1, int(round(args.train_frac * len(tr_in))))
        tr_in = rng.choice(tr_in, size=n_keep, replace=False)
    te_idx = np.where(te_mask)[0]

    y = df[ycol].to_numpy(dtype=float)
    Xraw = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    xs = StandardScaler().fit(Xraw[tr_in])
    ys = StandardScaler().fit(y[tr_in].reshape(-1, 1))
    Xs = xs.transform(Xraw)
    ysc = ys.transform(y.reshape(-1, 1)).ravel()

    from chemprop import featurizers
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    smiles = df["smiles"].to_numpy()
    tr_dset = make_dataset(smiles[tr_in], Xs[tr_in], ysc[tr_in], featurizer)
    val_dset = make_dataset(smiles[val_idx], Xs[val_idx], ysc[val_idx], featurizer)
    # eval_idx: report on frozen test (honest final number) OR on the early-stopping val
    # fold (hyperparameter selection -- test never loaded). The val metric is mildly
    # optimistic (early stopping peeked at it) but that optimism is uniform across configs,
    # so the config RANKING is valid; re-run the winner with --eval-on test for the number.
    eval_idx = te_idx if args.eval_on == "test" else val_idx
    eval_dset = make_dataset(smiles[eval_idx], Xs[eval_idx], None, featurizer)

    model, trainer = train_one(tr_dset, val_dset, params, args.seed, n_xd=Xraw.shape[1])
    pred_sc = predict(model, trainer, eval_dset, params)
    pred = ys.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    y_te = y[eval_idx]

    result = {
        "which": args.which, "target": ycol, "n": len(df), "n_xd": int(Xraw.shape[1]),
        "eval_on": args.eval_on, "params": params,
        "n_train": int(len(tr_in)), "n_val": int(len(val_idx)), "n_test": int(len(te_idx)),
        "n_eval": int(len(eval_idx)),
        "MAE": float(mean_absolute_error(y_te, pred)),
        "RMSE": float(root_mean_squared_error(y_te, pred)),
        "R2": float(r2_score(y_te, pred)),
        "spearman_rho": float(spearmanr(y_te, pred).correlation),
    }
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")

    if args.pred_out:
        pd.DataFrame({"id": df["id"].to_numpy()[eval_idx], "y_true": y_te, "y_pred": pred}
                     ).to_csv(args.pred_out, index=False)
        print(f"wrote {args.pred_out}")

    if args.save_checkpoint:
        import torch
        ckpt_path = Path(args.save_checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "which": args.which,
            "target_col": ycol,
            "feats": feats,
            "n_xd": int(Xraw.shape[1]),
            "params": params,
            "seed": args.seed,
            "split_file": str(args.split_file) if args.split_file else None,
            "eval_on": args.eval_on,
            "model_state_dict": model.state_dict(),
            "x_scaler": xs,
            "y_scaler": ys,
            "result": result,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"wrote checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
