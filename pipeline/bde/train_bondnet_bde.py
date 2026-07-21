#!/usr/bin/env python
"""Phase-1 baseline B5 (BDE_prediction.md section 六): "BonDNet-style" reaction-difference
D-MPNN -- a *learned graph-embedding* analogue of the B3 D-SPOC descriptor-difference
baseline:

    h_rxn = h(fragment1) + h(fragment2) - h(parent)
    BDE_pred = FFN(h_rxn)

where h(.) is a shared (weight-tied) BondMessagePassing encoder + mean-pool, called once
per graph (parent / frag1 / frag2) each batch. This is chemprop's architecture but NOT
chemprop's `models.MPNN` class, which assumes exactly one graph per datapoint -- chemprop
v2's multi-component support concatenates encodings (built for solvent+solute-style
inputs), not subtracts them, so the reaction-difference combinator here is a small custom
Lightning module built directly on chemprop's public `nn.BondMessagePassing` /
`nn.MeanAggregation` / `data.BatchMolGraph` building blocks.

Contrast with train_gnn_bde.py (B4): B4 encodes ONLY the parent graph and lets the model
implicitly learn "this graph's BDE at its one fixed bond position." B5 instead makes the
bond-breaking process explicit in the architecture (mirrors BDE = E(fragA)+E(fragB)-E(AB)
the same way B3 does with descriptors instead of learned embeddings) -- the doc's stated
reason B5 exists alongside B4.

Reuses fragment1/fragment2 SMILES from pipeline/compute/calc_bde_alfabet.py's output
(same as train_dspoc_baseline.py) -- no separate fragmentation step.

Usage:
  python train_bondnet_bde.py --which aldehydes \
      --alfabet-csv .../aldehydes_bde_alfabet.csv --out /tmp/bondnet_ald.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from qc import qc_filter
from splits import molecule_cold_split

RDLogger.DisableLog("rdApp.*")
H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

DEFAULT_PARAMS = dict(depth=4, d_h=300, ffn_hidden=300, dropout=0.0, batch_size=64,
                       max_epochs=120, patience=20, val_frac=0.1, lr=1e-3)


class ReactionDataset(Dataset):
    def __init__(self, parents, frag1s, frag2s, y, featurizer):
        self.parents, self.frag1s, self.frag2s = parents, frag1s, frag2s
        self.y = y
        self.featurizer = featurizer

    def __len__(self):
        return len(self.parents)

    def __getitem__(self, i):
        mg_p = self.featurizer(Chem.MolFromSmiles(self.parents[i]))
        mg_1 = self.featurizer(Chem.MolFromSmiles(self.frag1s[i]))
        mg_2 = self.featurizer(Chem.MolFromSmiles(self.frag2s[i]))
        yi = None if self.y is None else float(self.y[i])
        return mg_p, mg_1, mg_2, yi


def make_collate():
    from chemprop.data import BatchMolGraph

    def collate(batch):
        mg_p, mg_1, mg_2, y = zip(*batch)
        bmg_p, bmg_1, bmg_2 = BatchMolGraph(mg_p), BatchMolGraph(mg_1), BatchMolGraph(mg_2)
        y_t = None if y[0] is None else torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        return bmg_p, bmg_1, bmg_2, y_t
    return collate


class BondNetModule:
    """Plain (non-Lightning) wrapper: kept simple since this is a bespoke 3-graph model,
    not something chemprop's Trainer abstractions are built for."""

    def __init__(self, params, device):
        from chemprop import nn as cpnn
        self.device = device
        self.mp = cpnn.BondMessagePassing(depth=params["depth"], d_h=params["d_h"]).to(device)
        self.agg = cpnn.MeanAggregation().to(device)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(params["d_h"], params["ffn_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(params["dropout"]),
            torch.nn.Linear(params["ffn_hidden"], 1),
        ).to(device)

    def parameters(self):
        return list(self.mp.parameters()) + list(self.agg.parameters()) + \
               list(self.ffn.parameters())

    def encode(self, bmg):
        bmg.to(self.device)  # BatchMolGraph.to() mutates in place, no return value
        return self.agg(self.mp(bmg), bmg.batch)

    def forward(self, bmg_p, bmg_1, bmg_2):
        h_rxn = self.encode(bmg_1) + self.encode(bmg_2) - self.encode(bmg_p)
        return self.ffn(h_rxn)

    def train_mode(self):
        self.mp.train(); self.agg.train(); self.ffn.train()

    def eval_mode(self):
        self.mp.eval(); self.agg.eval(); self.ffn.eval()


def run_epoch(model, loader, optim=None):
    training = optim is not None
    model.train_mode() if training else model.eval_mode()
    total_loss, n = 0.0, 0
    all_pred = []
    with torch.set_grad_enabled(training):
        for bmg_p, bmg_1, bmg_2, y in loader:
            pred = model.forward(bmg_p, bmg_1, bmg_2)
            all_pred.append(pred.detach().cpu().numpy().ravel())
            if y is not None:
                y = y.to(model.device)
                loss = torch.nn.functional.mse_loss(pred, y)
                if training:
                    optim.zero_grad(); loss.backward(); optim.step()
                total_loss += loss.item() * len(y); n += len(y)
    return (total_loss / max(n, 1)), np.concatenate(all_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--alfabet-csv", required=True)
    ap.add_argument("--target", choices=["bde", "bdfe"], default="bde")
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
                          "(2026-07-17). 'validation' rows fold into train, only 'test' held out.")
    args = ap.parse_args()
    params = dict(DEFAULT_PARAMS)
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
    torch.manual_seed(args.seed)

    frags = pd.read_csv(args.alfabet_csv, dtype={"id": str})
    frags = frags.rename(columns={"smiles_canonical": "smiles"}).drop_duplicates("id")

    labels = pd.read_csv(H / f"{args.which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    ycol = f"{args.target}_gxtb_kcal"
    labels = labels.dropna(subset=[ycol]).drop_duplicates("id")
    labels = labels[qc_filter(labels[ycol])]

    id_cols = ["id", "donor_id"] if args.which == "products" else ["id"]
    extra = pd.read_csv(H / f"{args.which}_all.csv", usecols=id_cols, dtype=str,
                         keep_default_na=False) if args.which == "products" else None

    df = labels.merge(frags, on="id", how="inner")
    if extra is not None:
        df = df.merge(extra, on="id", how="inner")
    print(f"{args.which}: {len(df)} rows with BDE label + ALFABET fragment pair", flush=True)

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
    rng = np.random.default_rng(args.seed)
    tr_idx = np.where(tr_mask)[0]
    perm = rng.permutation(tr_idx)
    n_val = max(1, int(round(params["val_frac"] * len(perm))))
    val_idx, tr_in = perm[:n_val], perm[n_val:]
    te_idx = np.where(te_mask)[0]
    print(f"train={len(tr_in)}  val={len(val_idx)}  test={len(te_idx)}  cold on '{split_col}'",
          flush=True)

    y = df[ycol].to_numpy(dtype=float)
    ys = StandardScaler().fit(y[tr_in].reshape(-1, 1))
    ysc = ys.transform(y.reshape(-1, 1)).ravel()

    from chemprop import featurizers
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    p, f1, f2 = df["smiles"].to_numpy(), df["fragment1"].to_numpy(), df["fragment2"].to_numpy()
    collate = make_collate()

    def loader_for(idx, y_arr, shuffle):
        ds = ReactionDataset(p[idx], f1[idx], f2[idx], y_arr[idx] if y_arr is not None else None,
                              featurizer)
        return DataLoader(ds, batch_size=params["batch_size"], shuffle=shuffle,
                           collate_fn=collate, num_workers=0)

    tr_loader = loader_for(tr_in, ysc, True)
    val_loader = loader_for(val_idx, ysc, False)
    te_loader = loader_for(te_idx, None, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BondNetModule(params, device)
    optim = torch.optim.Adam(model.parameters(), lr=params["lr"])

    best_val, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(params["max_epochs"]):
        tr_loss, _ = run_epoch(model, tr_loader, optim)
        val_loss, _ = run_epoch(model, val_loader, None)
        print(f"epoch {epoch} train_mse={tr_loss:.4f} val_mse={val_loss:.4f}", flush=True)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: {kk: vv.clone() for kk, vv in v.state_dict().items()}
                           for k, v in [("mp", model.mp), ("agg", model.agg), ("ffn", model.ffn)]}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= params["patience"]:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.mp.load_state_dict(best_state["mp"])
        model.agg.load_state_dict(best_state["agg"])
        model.ffn.load_state_dict(best_state["ffn"])

    _, pred_sc = run_epoch(model, te_loader, None)
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
