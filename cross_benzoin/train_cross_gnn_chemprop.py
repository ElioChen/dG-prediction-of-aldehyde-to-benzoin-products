#!/usr/bin/env python
"""Chemprop (D-MPNN) comparison point for cross-benzoin's dG GNN leg.

User-requested (2026-09-16): "try chemprop on the main project." The deployed
champion's GNN (`train_cross_gnn.py`'s TripleGNN) is a hand-built PyTorch
Geometric model (3 separate GINEConv encoders for product/donor/acceptor,
concatenated + a donor-acceptor asymmetry term + the QM-scalar channel, see
`cross_benzoin/gnn_architectures.py`'s header for why the two projects
diverged: BDE's default was chemprop's D-MPNN, cross's own was never built on
chemprop). This is the first attempt at that comparison for the dG task
itself (chemprop is already the champion architecture one sub-project over,
`pipeline/bde/train_gnn_hybrid_bde.py` -- see [[bde-phase1-status]]).

Architecture: chemprop v2's native multi-component support
(`MulticomponentMessagePassing`, 3 separate `BondMessagePassing` blocks,
`shared=False` to match TripleGNN's 3-separate-encoder design) over
[product, donor, acceptor] SMILES, `MeanAggregation` readout, concatenated
fingerprint + the SAME 257-feature champion QM/mordred/rdkit schema as `x_d`
(chemprop's convention: x_d/y are read from the FIRST component's dataset
only -- collate_multicomponent -- so both are attached to the product
component here). No donor-acceptor asymmetry term (chemprop's stock FFN
readout has no hook for it without subclassing further; noted as a
simplification, not attempted this pass).

Same Delta-learning setup, same env-overridable target/baseline columns as
`train_cross_gnn.py` -- but split via the table's own `new_scaffold_split`
column (train_mask = 'train' only, i.e. "clean-train", same convention as
`homo_cross_joint_tabular.py`/`train_scaffold_disjoint.py`), NOT
`train_cross_delta.pair_split_labels()` the way `train_cross_gnn.py` itself
does: that function reads `SPLIT_MAP` = the now-retired candidates_v3 split
parquet (`data/cross_benzoin/_archive/candidates_v3/inchikey_split_map.parquet`),
which does not exist on disk (never restored post-2026-07-purge, per
LAB_JOURNAL's 2026-09-15 candidates_v3 retirement entry) -- calling it today
returns None unconditionally, which would silently route ALL rows into
`train_extra` (empty val/test masks). Real, currently-latent landmine in
`train_cross_gnn.py` for its next re-run, found while building this script;
not fixed here (out of scope), flagged to the user instead. `new_scaffold_split`
is what `train_scaffold_disjoint.py` (the actually-deployed tabular trainer)
already uses instead, for exactly this reason, so this script's holdout is
directly comparable to the deployed champion's own frozen test split (n=448).

Env: /home/schen3/venv/bde_gnn (torch 2.13+cu130, chemprop 2.2.0, lightning --
survived the 2026-07 purge / was rebuilt post-purge; NOT the same env as the
champion's own GNN training, which uses /home/schen3/venv/nequip's PyG stack).

Usage
  # CPU correctness smoke test (~200 rows, 2 epochs, ~1 min):
  /home/schen3/venv/bde_gnn/bin/python cross_benzoin/train_cross_gnn_chemprop.py \
      --table data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet \
      --feature-list data/cross_benzoin/feature_list_257_no_nCHO_v2.json \
      --outdir /tmp/chemprop_smoke --smoke

  # real run (GPU, via sbatch submit_cross_gnn_chemprop.sh):
  CB_TARGET_COL=dG_r2scan_kcal CB_BASELINE_COL=dG_b973c_kcal \
  /home/schen3/venv/bde_gnn/bin/python cross_benzoin/train_cross_gnn_chemprop.py \
      --table data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet \
      --feature-list data/cross_benzoin/feature_list_257_no_nCHO_v2.json \
      --outdir data/cross_benzoin/cross_round10/gnn_chemprop_10rounds_b973c_v1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def build_component_dataset(smiles, x_d, y, featurizer):
    from chemprop import data
    dps = []
    for i, smi in enumerate(smiles):
        yi = None if y is None else np.array([y[i]], dtype=float)
        xdi = None if x_d is None else np.asarray(x_d[i], dtype=float)
        dps.append(data.MoleculeDatapoint.from_smi(str(smi), y=yi, x_d=xdi))
    return data.MoleculeDataset(dps, featurizer)


def build_mcmpnn(n_xd: int, depth: int, d_h: int, ffn_hidden: int, ffn_layers: int, dropout: float):
    from chemprop import models, nn as cpnn
    blocks = [cpnn.BondMessagePassing(depth=depth, d_h=d_h) for _ in range(3)]
    mp = cpnn.MulticomponentMessagePassing(blocks=blocks, n_components=3, shared=False)
    agg = cpnn.MeanAggregation()
    out_dim = sum(b.output_dim for b in blocks)
    ffn = cpnn.RegressionFFN(input_dim=out_dim + n_xd, hidden_dim=ffn_hidden,
                              n_layers=ffn_layers, dropout=dropout)
    return models.MulticomponentMPNN(mp, agg, ffn, batch_norm=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--feature-list", type=Path, required=True,
                     help="champion feature_list.json (257-feat schema v2)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--ffn-hidden", type=int, default=300)
    ap.add_argument("--ffn-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                     help="~300-row subset, CPU, 2 epochs -- API/shape correctness check only, not a real result")
    args = ap.parse_args()

    t0 = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    target_col = os.environ.get("CB_TARGET_COL", "dG_orca_kcal")
    baseline_col = os.environ.get("CB_BASELINE_COL", "dG_gxtb_kcal")
    print(f"target={target_col} baseline={baseline_col}", flush=True)

    df = pd.read_parquet(args.table)
    feats = json.loads(args.feature_list.read_text())
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"table missing {len(missing)} champion feature cols, e.g. {missing[:5]}")

    if "new_scaffold_split" not in df.columns:
        raise SystemExit("table has no new_scaffold_split column")
    print(f"rows: {len(df)} ({df['new_scaffold_split'].value_counts().to_dict()})", flush=True)

    train_mask = (df["new_scaffold_split"] == "train").to_numpy()
    val_mask = (df["new_scaffold_split"] == "validation").to_numpy()
    test_mask = (df["new_scaffold_split"] == "test").to_numpy()

    if args.smoke:
        rng = np.random.default_rng(0)
        tr_idx = rng.choice(np.where(train_mask)[0], size=min(200, int(train_mask.sum())), replace=False)
        va_idx = np.where(val_mask)[0][:50]
        te_idx = np.where(test_mask)[0][:50]
        df = df.iloc[np.concatenate([tr_idx, va_idx, te_idx])].reset_index(drop=True)
        train_mask = (df["new_scaffold_split"] == "train").to_numpy()
        val_mask = (df["new_scaffold_split"] == "validation").to_numpy()
        test_mask = (df["new_scaffold_split"] == "test").to_numpy()
        print(f"SMOKE MODE: subsampled to {len(df)} rows", flush=True)

    # QM-scalar x_d: median-impute + standardize, fit on train rows only (matches train_cross_gnn.py)
    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    med = Xdf[train_mask].median(numeric_only=True)
    Xz = Xdf.fillna(med).fillna(0.0)
    qm_mean = Xz[train_mask].mean().to_numpy()
    qm_std = Xz[train_mask].std().replace(0, 1).to_numpy()
    qm_std = np.where(qm_std == 0, 1.0, qm_std)
    qmz = ((Xz.to_numpy() - qm_mean) / qm_std).astype(np.float32)

    delta = (df[target_col] - df[baseline_col]).to_numpy(dtype=np.float64)
    ym, ysd = float(delta[train_mask].mean()), float(delta[train_mask].std())
    deltaz = ((delta - ym) / ysd).astype(np.float32)

    from chemprop import data, featurizers
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def make_multi(mask):
        idx = np.where(mask)[0]
        prod_ds = build_component_dataset(df["smiles"].iloc[idx], qmz[idx], deltaz[idx], featurizer)
        don_ds = build_component_dataset(df["donor_smiles"].iloc[idx], None, None, featurizer)
        acc_ds = build_component_dataset(df["acceptor_smiles"].iloc[idx], None, None, featurizer)
        return data.MulticomponentDataset([prod_ds, don_ds, acc_ds]), idx

    print("building graphs (product + donor + acceptor)...", flush=True)
    tr_dset, tr_idx = make_multi(train_mask)
    val_dset, val_idx = make_multi(val_mask)
    te_dset, te_idx = make_multi(test_mask)
    print(f"train {len(tr_idx)}  val {len(val_idx)}  test {len(te_idx)}", flush=True)

    tr_loader = data.build_dataloader(tr_dset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = data.build_dataloader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te_loader = data.build_dataloader(te_dset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_mcmpnn(qmz.shape[1], args.depth, args.hidden, args.ffn_hidden, args.ffn_layers, args.dropout)

    from lightning import pytorch as pl
    pl.seed_everything(args.seed, workers=True)
    cbs = [pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)]
    max_epochs = 2 if args.smoke else args.max_epochs
    accelerator = "cpu" if args.smoke else "auto"
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=1,
                          enable_progress_bar=False, enable_checkpointing=False,
                          logger=False, callbacks=cbs, deterministic=False)
    trainer.fit(model, tr_loader, val_loader)

    def eval_split(loader, idx):
        preds = trainer.predict(model, loader)
        yhat_z = np.concatenate([p.numpy().reshape(-1) for p in preds])
        yhat_delta = yhat_z * ysd + ym
        yhat = df[baseline_col].to_numpy()[idx] + yhat_delta
        ytrue = df[target_col].to_numpy()[idx]
        mae = float(np.abs(ytrue - yhat).mean())
        return mae, yhat, ytrue

    val_mae = eval_split(val_loader, val_idx)[0]
    te_mae, te_yhat, te_ytrue = eval_split(te_loader, te_idx)
    print(f"validation MAE={val_mae:.4f}", flush=True)
    print(f"test MAE={te_mae:.4f}  (n={len(te_idx)})", flush=True)

    torch_state_path = args.outdir / "models" / "gnn_chemprop_state.pt"
    torch_state_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(model.state_dict(), torch_state_path)

    out = {
        "built_by": "cross_benzoin/train_cross_gnn_chemprop.py",
        "smoke": bool(args.smoke),
        "table": str(args.table), "target_col": target_col, "baseline_col": baseline_col,
        "n_train": len(tr_idx), "n_val": len(val_idx), "n_test": len(te_idx),
        "n_xd": int(qmz.shape[1]), "ym": ym, "ysd": ysd,
        "params": {"depth": args.depth, "hidden": args.hidden, "ffn_hidden": args.ffn_hidden,
                   "ffn_layers": args.ffn_layers, "dropout": args.dropout,
                   "batch_size": args.batch_size, "max_epochs": max_epochs, "seed": args.seed},
        "val_mae": val_mae, "test_mae": te_mae,
        "cross_reference_champion_gnn": {"best_single_seed_mae": 0.556, "4seed_avg_mae": 0.531,
                                          "blend_mae": 0.528},
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    (args.outdir / "result.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n-> {args.outdir / 'result.json'}  ({out['runtime_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
