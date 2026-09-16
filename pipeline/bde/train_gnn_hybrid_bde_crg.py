#!/usr/bin/env python
"""CRG (condensed/reaction-graph-style) ablation of the B6 champion (user-requested
2026-09-16, "BDE 等工作也可以尝试" -- CRG applied to BDE, following the same idea
already validated for cross-benzoin's dG GNN).

B6 (`train_gnn_hybrid_bde.py`) predicts a bond's dissociation (free) energy from the
WHOLE molecule's graph + local descriptors, relying entirely on molecule-class
homogeneity to know which bond is meant (every aldehyde row is asked about ITS OWN
formyl C-H, every product row about ITS OWN ketC-carbC -- "no explicit bond-marking
is needed", per that file's own docstring). This script tests whether explicitly
marking the target bond as a graph feature -- the core CRG idea -- helps, using
chemprop's own native extension hooks (`extra_bond_fdim`/`E_f`) so this is a precise,
architecture-preserving ablation: same BondMessagePassing/MeanAggregation/
RegressionFFN, same local-descriptor x_d, same data/split, only difference is one
extra bond feature.

Scope: **products only** (Task B, ketC-carbC). Task A (aldehyde formyl C-H) has no
graph edge to mark under this project's implicit-H representation (H isn't a node) --
the dissociating bond terminates at an atom that doesn't exist in the graph, so
"mark the bond" has no direct analogue there (a node-level flag on CHO_C would be a
different, weaker intervention, not attempted here). Products' ketC-carbC bond IS
between two heavy atoms, so it can be marked exactly like cross-benzoin's dG-CRG
marks the new coupling bond -- reuses `crg_builder.find_reactive_core` (the same
SMARTS-based reactive-core locator built for that, validated 97.5%+ unique-match
rate) to find ketC/carbC, then chemprop's `extra_bond_fdim=1` + per-datapoint `E_f`
to flag that one bond.

Usage
  /home/schen3/venv/bde_gnn/bin/python pipeline/bde/train_gnn_hybrid_bde_crg.py \
      --depth 4 --message-hidden 500 --eval-on test --seed 0 \
      --split-file data/cross_benzoin/homo_v6/products_scaffold_split.csv \
      --out /tmp/b6_crg_products_seed0.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc import norm_id, qc_filter
from splits import molecule_cold_split
from train_gnn_hybrid_bde import (  # noqa: E402
    LOCAL_FEATURES, DEFAULT_PARAMS, MP_CLASSES, AGG_CLASSES, H, train_one, predict,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cross_benzoin"))
from crg_builder import find_reactive_core  # noqa: E402


def target_bond_features(smi: str) -> np.ndarray | None:
    """Per-bond E_f array (n_bonds x 1), 1.0 on the ketC-carbC bond, else 0. None if
    the reactive core can't be uniquely located (same graceful-drop convention as
    crg_builder.build_crg)."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None or mol.GetNumBonds() == 0:
        return None
    core = find_reactive_core(mol)
    if core is None:
        return None
    ket_c, _ket_o, carb_c, _hyd_o = core
    bond = mol.GetBondBetweenAtoms(ket_c, carb_c)
    if bond is None:
        return None
    e_f = np.zeros((mol.GetNumBonds(), 1), dtype=np.float32)
    e_f[bond.GetIdx(), 0] = 1.0
    return e_f


def build_mpnn_crg(n_xd: int, params: dict):
    from chemprop import models, nn as cpnn
    mp_cls = getattr(cpnn, MP_CLASSES[params.get("message_passing", "bond")])
    mp = mp_cls(depth=params["depth"], d_h=params["message_hidden"], d_e=15)  # 14 + 1 extra
    mp_out_dim = mp.output_dim if hasattr(mp, "output_dim") else mp.output_dims[0]
    agg_name = params.get("aggregation", "mean")
    agg = cpnn.AttentiveAggregation(output_size=mp_out_dim) if agg_name == "attentive" \
        else getattr(cpnn, AGG_CLASSES[agg_name])()
    ffn = cpnn.RegressionFFN(input_dim=mp_out_dim + n_xd, hidden_dim=params["ffn_hidden"],
                              n_layers=params["ffn_layers"], dropout=params["dropout"])
    return models.MPNN(mp, agg, ffn, batch_norm=True)


def make_dataset_crg(smiles, X, y, e_fs, featurizer):
    from chemprop import data
    dps = []
    for i, smi in enumerate(smiles):
        yi = None if y is None else np.array([y[i]], dtype=float)
        dps.append(data.MoleculeDatapoint.from_smi(smi, y=yi, x_d=np.asarray(X[i], dtype=float),
                                                     E_f=e_fs[i]))
    return data.MoleculeDataset(dps, featurizer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["bde", "bdfe"], default="bde")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-out", default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--message-hidden", type=int, default=None)
    ap.add_argument("--ffn-hidden", type=int, default=None)
    ap.add_argument("--ffn-layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--eval-on", choices=["test", "val"], default="test")
    ap.add_argument("--split-file", type=Path, default=None)
    ap.add_argument("--no-mark", action="store_true",
                     help="ablation control: keep the extra_bond_fdim=1 channel (identical "
                          "param count) but always zero it out -- isolates the marking's "
                          "information content from the row-subset/architecture-capacity match")
    args = ap.parse_args()

    which = "products"
    params = dict(DEFAULT_PARAMS)
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
    for k, v in [("depth", args.depth), ("message_hidden", args.message_hidden),
                 ("ffn_hidden", args.ffn_hidden), ("ffn_layers", args.ffn_layers),
                 ("dropout", args.dropout), ("batch_size", args.batch_size)]:
        if v is not None:
            params[k] = v
    params["message_passing"] = "bond"
    params["aggregation"] = "mean"

    feats = LOCAL_FEATURES[which]
    labels = pd.read_csv(H / f"{which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    labels["id"] = norm_id(labels["id"])
    ycol = f"{args.target}_gxtb_kcal"
    labels = labels.dropna(subset=[ycol]).drop_duplicates("id")
    labels = labels[qc_filter(labels[ycol])]

    id_cols = ["id", "donor_id", "smiles", "error"] + feats
    mol = pd.read_csv(H / f"{which}_all.csv", usecols=id_cols, dtype=str,
                       keep_default_na=False, low_memory=False)
    mol = mol[mol["error"] == ""]
    mol["id"] = norm_id(mol["id"])
    mol["donor_id"] = norm_id(mol["donor_id"])
    for c in feats:
        mol[c] = pd.to_numeric(mol[c], errors="coerce")

    df = labels.merge(mol, on="id", how="inner").dropna(subset=feats, how="all")
    df = df[df["smiles"] != ""].reset_index(drop=True)

    print(f"{which}: {len(df)} rows with BDE label + smiles + local-3D descriptors "
          f"before CRG filtering", flush=True)
    e_f_list = [target_bond_features(s) for s in df["smiles"]]
    keep = np.array([e is not None for e in e_f_list])
    print(f"CRG target-bond located: {keep.sum()}/{len(df)} "
          f"({(~keep).sum()} dropped -- ambiguous/no reactive-core match)", flush=True)
    df = df[keep].reset_index(drop=True)
    e_f_list = [e for e, k in zip(e_f_list, keep) if k]
    if args.no_mark:
        e_f_list = [np.zeros_like(e) for e in e_f_list]

    if args.n is not None:
        idx = np.random.default_rng(args.seed).choice(len(df), size=min(args.n, len(df)), replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        e_f_list = [e_f_list[i] for i in idx]

    split_col = "donor_id"
    if args.split_file is not None:
        read = pd.read_csv if args.split_file.suffix == ".csv" else pd.read_parquet
        sf = read(args.split_file)[["id", "scaffold_split"]]
        sf["id"] = norm_id(sf["id"])
        df = df.merge(sf, on="id", how="left")
        n_unmatched = df["scaffold_split"].isna().sum()
        if n_unmatched:
            print(f"WARN: {n_unmatched}/{len(df)} rows have no scaffold_split assignment "
                  f"(dropped)", flush=True)
        keep2 = df["scaffold_split"].notna().to_numpy()
        e_f_list = [e for e, k in zip(e_f_list, keep2) if k]
        df = df[keep2].reset_index(drop=True)
        split = df["scaffold_split"].replace({"validation": "train"})
        print(f"{which}: using scaffold-disjoint --split-file {args.split_file.name}", flush=True)
    else:
        split = molecule_cold_split(df[split_col], test_frac=args.test_frac, seed=args.seed)
    tr_mask, te_mask = (split == "train").to_numpy(), (split == "test").to_numpy()
    print(f"{which}: n={len(df)}  train={tr_mask.sum()}  test={te_mask.sum()}  cold on '{split_col}'",
          flush=True)

    rng = np.random.default_rng(args.seed)
    tr_idx = np.where(tr_mask)[0]
    perm = rng.permutation(tr_idx)
    n_val = max(1, int(round(params["val_frac"] * len(perm))))
    val_idx, tr_in = perm[:n_val], perm[n_val:]
    te_idx = np.where(te_mask)[0]

    y = df[ycol].to_numpy(dtype=float)
    Xraw = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    xs = StandardScaler().fit(Xraw[tr_in])
    ys = StandardScaler().fit(y[tr_in].reshape(-1, 1))
    Xs = xs.transform(Xraw)
    ysc = ys.transform(y.reshape(-1, 1)).ravel()

    from chemprop import featurizers
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_bond_fdim=1)
    smiles = df["smiles"].to_numpy()
    tr_dset = make_dataset_crg(smiles[tr_in], Xs[tr_in], ysc[tr_in],
                                [e_f_list[i] for i in tr_in], featurizer)
    val_dset = make_dataset_crg(smiles[val_idx], Xs[val_idx], ysc[val_idx],
                                 [e_f_list[i] for i in val_idx], featurizer)
    eval_idx = te_idx if args.eval_on == "test" else val_idx
    eval_dset = make_dataset_crg(smiles[eval_idx], Xs[eval_idx], None,
                                  [e_f_list[i] for i in eval_idx], featurizer)

    model = build_mpnn_crg(Xraw.shape[1], params)
    model, trainer = train_one(tr_dset, val_dset, params, args.seed, n_xd=Xraw.shape[1],
                                model_override=model)
    pred_sc = predict(model, trainer, eval_dset, params)
    pred = ys.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    y_te = y[eval_idx]

    result = {
        "built_by": "pipeline/bde/train_gnn_hybrid_bde_crg.py",
        "no_mark": bool(args.no_mark),
        "which": which, "target": ycol, "n": len(df), "n_xd": int(Xraw.shape[1]),
        "n_dropped_no_core_match": int((~keep).sum()),
        "eval_on": args.eval_on, "params": params, "seed": args.seed,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
