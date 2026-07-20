#!/usr/bin/env python
"""Phase-1 baseline B4 (BDE_prediction.md, section 六): a D-MPNN predicting the project's
own g-xTB BDE directly from the 2D graph -- aldehyde formyl C-H BDE from the aldehyde
graph, product ketC-carbC BDE from the product graph.

Not literally "bond-centered" in the ALFABET sense (no per-bond marking token): both
targets are always the SAME semantic bond position (the one formyl C-H / the one new
ketC-carbC bond), fixed by the molecule class, so there is exactly one candidate bond per
graph and the model only has to learn "this graph's BDE at that fixed position" as a
graph-level regression -- no atom/bond marking needed, unlike ALFABET which predicts over
every bond of an arbitrary input.

Reuses the MPNN architecture (BondMessagePassing + MeanAggregation + RegressionFFN) from
pipeline/gnn/gnn_core.py, but WITHOUT its dG_gxtb Δ-learning framing (no x_d, no dG_xtb
baseline to add back -- this predicts the label directly) and WITHOUT its row-level
RepeatedKFold (replaced with pipeline/bde/splits.py:molecule_cold_split, molecule-level
cold start, consistent with the rest of this Phase-1 baseline work).

Deliberately in its OWN isolated env (envs/bde_gnn) rather than the existing envs/gnn --
that env's stdlib (lib/python3.12/encodings/) was found empty/corrupted while this task
was being set up (unrelated to this session's own work; flagged separately, not touched).

Usage (smoke test on a subset):
  python train_gnn_bde.py --which aldehydes --n 500 --max-epochs 5 --out /tmp/gnn_bde_smoke.json
Full run (submit via SLURM, GPU node -- see submit_bde_gnn.sh):
  python train_gnn_bde.py --which aldehydes --out /tmp/gnn_bde_ald.json
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

DEFAULT_PARAMS = dict(depth=4, message_hidden=300, ffn_hidden=300, ffn_layers=2,
                       dropout=0.0, batch_size=64, max_epochs=120, patience=20,
                       val_frac=0.1)


def build_mpnn(params):
    from chemprop import models, nn as cpnn
    mp = cpnn.BondMessagePassing(depth=params["depth"], d_h=params["message_hidden"])
    agg = cpnn.MeanAggregation()
    ffn = cpnn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=params["ffn_hidden"],
                              n_layers=params["ffn_layers"], dropout=params["dropout"])
    return models.MPNN(mp, agg, ffn, batch_norm=True)


def make_dataset(smiles, y, featurizer):
    from chemprop import data
    dps = [data.MoleculeDatapoint.from_smi(smi, y=None if y is None else np.array([y[i]]))
           for i, smi in enumerate(smiles)]
    return data.MoleculeDataset(dps, featurizer)


def train_one(tr_dset, val_dset, params, seed):
    from chemprop import data
    from lightning import pytorch as pl
    pl.seed_everything(seed, workers=True)
    model = build_mpnn(params)
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
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-out", default=None,
                     help="optional CSV of per-row test predictions (id,y_true,y_pred), "
                          "for ensembling with other baselines sharing the same split")
    ap.add_argument("--split-file", type=Path, default=None,
                     help="if set, use this precomputed id->scaffold_split CSV/parquet "
                          "INSTEAD of molecule_cold_split -- genuinely scaffold-disjoint "
                          "(2026-07-17), not just molecule-disjoint. 'validation' rows (if "
                          "present) fold into train; only 'test' is held out.")
    args = ap.parse_args()
    params = dict(DEFAULT_PARAMS)
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs

    labels = pd.read_csv(H / f"{args.which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    ycol = f"{args.target}_gxtb_kcal"
    labels = labels.dropna(subset=[ycol]).drop_duplicates("id")
    # A handful of rows have wildly divergent SCF energies (e.g. 1.1e6 kcal/mol) that
    # were never NaN-filtered upstream -- clip to a physically sane BDE/BDFE window.
    n_before = len(labels)
    labels = labels[qc_filter(labels[ycol])]
    print(f"QC: dropped {n_before - len(labels)}/{n_before} rows outside [20,200] kcal/mol")

    id_cols = ["id", "smiles"] if args.which == "aldehydes" else ["id", "donor_id", "smiles"]
    mol = pd.read_csv(H / f"{args.which}_all.csv", usecols=id_cols, dtype=str,
                       keep_default_na=False)
    df = labels.merge(mol, on="id", how="inner")
    df = df[df["smiles"] != ""].reset_index(drop=True)
    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    split_col = "id" if args.which == "aldehydes" else "donor_id"
    if args.split_file is not None:
        read = pd.read_csv if args.split_file.suffix == ".csv" else pd.read_parquet
        sf = read(args.split_file)[["id", "scaffold_split"]]
        sf["id"] = sf["id"].astype(str)
        df = df.merge(sf, on="id", how="left")
        n_unmatched = df["scaffold_split"].isna().sum()
        if n_unmatched:
            print(f"WARN: {n_unmatched}/{len(df)} rows unmatched in split file, dropped", flush=True)
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
    te_idx = np.where(te_mask)[0]

    y = df[ycol].to_numpy(dtype=float)
    ys = StandardScaler().fit(y[tr_in].reshape(-1, 1))
    ysc = ys.transform(y.reshape(-1, 1)).ravel()

    from chemprop import featurizers
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    smiles = df["smiles"].to_numpy()
    tr_dset = make_dataset(smiles[tr_in], ysc[tr_in], featurizer)
    val_dset = make_dataset(smiles[val_idx], ysc[val_idx], featurizer)
    te_dset = make_dataset(smiles[te_idx], None, featurizer)

    model, trainer = train_one(tr_dset, val_dset, params, args.seed)
    pred_sc = predict(model, trainer, te_dset, params)
    pred = ys.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    y_te = y[te_idx]

    result = {
        "which": args.which, "target": ycol, "n": len(df),
        "n_train": int(len(tr_in)), "n_val": int(len(val_idx)), "n_test": int(len(te_idx)),
        "MAE": float(mean_absolute_error(y_te, pred)),
        "RMSE": float(root_mean_squared_error(y_te, pred)),
        "R2": float(r2_score(y_te, pred)),
        "spearman_rho": float(spearmanr(y_te, pred).correlation),
    }
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")

    if args.pred_out:
        pd.DataFrame({"id": df["id"].to_numpy()[te_idx], "y_true": y_te, "y_pred": pred}
                     ).to_csv(args.pred_out, index=False)
        print(f"wrote {args.pred_out}")


if __name__ == "__main__":
    main()
