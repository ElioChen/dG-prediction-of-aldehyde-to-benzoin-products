#!/usr/bin/env python
"""
Phase 2 DGT-inspired ablation: real pairwise 3D atom-atom distance attention (see
gnn_architectures.py's DistanceAttentionLayer/EncAttn3DGlobal/TripleGNNDistAttn) vs the
current production architecture, on the IDENTICAL geometry-matched row subset and split
as Phase 1 (gnn3d_train_and_compare.py) so results are directly comparable and any delta
is attributable to the architecture change alone.

--arch selects which model this run trains:
  attentive2d  -- current production architecture (TripleGNNAttn, plain 2D graphs)
  distattn     -- TripleGNNDistAttn, product encoder gets GINEConv topology + distance-
                  biased global self-attention (full pairwise atom-atom distances, not
                  just bonded pairs -- see module docstring in gnn_architectures.py)

Usage
  python cross_benzoin/gnn3d_train_and_compare_v2.py \
      --table data/cross_benzoin/cross_round9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet \
      --coords data/cross_benzoin/gnn3d/product_coords_r2348_r9.parquet \
      --champion-dir data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1 \
      --ensemble-path data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1/models/ensemble_scaffold_disjoint.joblib \
      --outdir data/cross_benzoin/gnn3d/distattn_v1 \
      --arch distattn --hidden 128 --layers 4 --lr 3e-4 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402
from train_cross_delta import TARGET_COL, BASELINE_COL  # noqa: E402
from train_cross_ensemble import MLPXGBEnsemble  # noqa: E402,F401
from train_cross_gnn import TripleData, af, bf, build_graphs, make_loader, evl, dev  # noqa: E402
from gnn_architectures import TripleGNNAttn, TripleGNNDistAttn  # noqa: E402

import __main__ as _main_mod  # noqa: E402
_main_mod.MLPXGBEnsemble = MLPXGBEnsemble


def graph_with_pos(smi: str, coords_flat: list[float]):
    """Same as train_cross_gnn.graph() (plain 2D topology, no bond-length injection --
    Phase 2 isolates the NEW distance-attention mechanism, doesn't stack Phase 1's edge
    hack on top) but also attaches `pos`: (n_heavy, 3) real coordinates, row-order-matched
    to Chem.MolFromSmiles(smi).GetAtoms() -- guaranteed by gnn3d_extract_product_coords.py."""
    m = Chem.MolFromSmiles(str(smi))
    if m is None or m.GetNumAtoms() == 0:
        return None
    n_heavy = m.GetNumAtoms()
    if len(coords_flat) != n_heavy * 3:
        return None  # defensive: schema drift, don't silently misalign
    x = torch.tensor([af(a) for a in m.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bf(b)
        ei += [[i, j], [j, i]]
        ea += [f, f]
    if not ei:
        ei = [[0, 0]]
        ea = [[0, 0, 0, 0, 0, 0]]
    pos = torch.tensor(coords_flat, dtype=torch.float).view(n_heavy, 3)
    return x, torch.tensor(ei).t().contiguous(), torch.tensor(ea, dtype=torch.float), pos


def build_graphs_dist(df: pd.DataFrame, qmz: np.ndarray, coords_map: dict[str, list[float]]) -> list[TripleData]:
    from train_cross_gnn import graph as graph2d
    pairs = []
    for k in range(len(df)):
        rid = str(df["id"].iloc[k])
        cf = coords_map.get(rid)
        if cf is None:
            continue
        gp = graph_with_pos(df["smiles"].iloc[k], cf)
        gd, ga = graph2d(df["donor_smiles"].iloc[k]), graph2d(df["acceptor_smiles"].iloc[k])
        if gp is None or gd is None or ga is None:
            continue
        d = TripleData()
        d.x_p, d.edge_index_p, d.edge_attr_p, d.x_p_pos = gp
        d.x_d, d.edge_index_d, d.edge_attr_d = gd
        d.x_a, d.edge_index_a, d.edge_attr_a = ga
        d.y = torch.tensor([float(df[TARGET_COL].iloc[k]) - float(df[BASELINE_COL].iloc[k])], dtype=torch.float)
        d.g0 = torch.tensor([float(df[BASELINE_COL].iloc[k])], dtype=torch.float)
        d.qm = torch.tensor(qmz[k], dtype=torch.float).view(1, -1)
        d.row_id = rid
        pairs.append(d)
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--coords", required=True, help="output of gnn3d_extract_product_coords.py")
    ap.add_argument("--champion-dir", required=True)
    ap.add_argument("--ensemble-path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arch", choices=["attentive2d", "distattn"], required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-dist-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-val-frac", type=float, default=0.1)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    feats = json.loads((Path(args.champion_dir) / "models" / "feature_list.json").read_text())
    df = pd.read_parquet(args.table)
    co_df = pd.read_parquet(args.coords)
    coords_map = dict(zip(co_df["id"], co_df["coords_flat"]))
    print(f"loaded table {len(df)} rows; coord cache {len(co_df)} rows", flush=True)

    df = df[df["id"].isin(coords_map.keys())].reset_index(drop=True)
    print(f"restricted to matched subset: {len(df)} rows", flush=True)

    df = df[df["new_scaffold_split"] != "mixed"].reset_index(drop=True)
    df["_split"] = df["new_scaffold_split"]
    if args.extra_val_frac > 0:
        rng = np.random.default_rng(args.seed)
        train_pairs_all = df.loc[df["_split"] == "train", "pair_key"].unique()
        n_extra_val = int(len(train_pairs_all) * args.extra_val_frac)
        val_pairs_extra = set(rng.choice(train_pairs_all, size=n_extra_val, replace=False))
        promote = (df["_split"] == "train") & df["pair_key"].isin(val_pairs_extra)
        df.loc[promote, "_split"] = "validation_extra"

    train_mask = (df["_split"] == "train").to_numpy()
    val_mask = df["_split"].isin(["validation", "validation_extra"]).to_numpy()
    test_mask = (df["_split"] == "test").to_numpy()
    print(f"final masks: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}", flush=True)

    Xdf = df[feats].apply(pd.to_numeric, errors="coerce")
    med = Xdf[train_mask].median(numeric_only=True)
    Xz = Xdf.fillna(med).fillna(0.0)
    qm_mean = Xz[train_mask].mean().to_numpy()
    qm_std = Xz[train_mask].std().replace(0, 1).to_numpy()
    qm_std = np.where(qm_std == 0, 1.0, qm_std)
    qmz = ((Xz.to_numpy() - qm_mean) / qm_std).astype(np.float32)

    t0 = time.time()
    if args.arch == "distattn":
        pairs = build_graphs_dist(df, qmz, coords_map)
    else:
        pairs = build_graphs(df, qmz)
    print(f"built {len(pairs)}/{len(df)} pairs ({time.time()-t0:.0f}s)", flush=True)

    id_to_split = dict(zip(df["id"].astype(str), df["_split"]))
    tr_pairs = [p for p in pairs if id_to_split.get(p.row_id) == "train"]
    va_pairs = [p for p in pairs if id_to_split.get(p.row_id) in ("validation", "validation_extra")]
    te_pairs = [p for p in pairs if id_to_split.get(p.row_id) == "test"]
    print(f"split: train={len(tr_pairs)} val={len(va_pairs)} test={len(te_pairs)}", flush=True)

    ys = torch.tensor([p.y.item() for p in tr_pairs])
    ym, ysd = ys.mean().item(), ys.std().item()
    for p in pairs:
        p.ystd = (p.y - ym) / ysd

    ad, bd = pairs[0].x_p.shape[1], pairs[0].edge_attr_a.shape[1]
    nqm = len(feats)
    torch.manual_seed(args.seed)
    if args.arch == "distattn":
        model = TripleGNNDistAttn(ad, bd, nqm, h=args.hidden, layers=args.layers,
                                   n_dist_layers=args.n_dist_layers, n_heads=args.n_heads).to(dev)
    else:
        model = TripleGNNAttn(ad, bd, nqm, h=args.hidden, layers=args.layers).to(dev)
    print(f"model={args.arch} params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    tl = make_loader(tr_pairs, args.batch_size, True)
    vl = make_loader(va_pairs, args.batch_size, False)
    best, best_state, pat = 1e9, None, 0
    t1 = time.time()
    for ep in range(args.max_epochs):
        model.train()
        for b in tl:
            b = b.to(dev)
            opt.zero_grad()
            loss = F.mse_loss(model(b), b.ystd)
            loss.backward()
            opt.step()
        vm = evl(model, vl, ym, ysd)[0]
        sch.step(vm)
        if vm < best - 1e-4:
            best, best_state, pat = vm, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if ep % 10 == 0 or pat == 0:
            print(f"  ep{ep:3d} val_MAE {vm:.3f} (best {best:.3f}) {time.time()-t1:.0f}s", flush=True)
        if pat >= args.patience:
            break
    model.load_state_dict(best_state)
    torch.save(best_state, out / "models" / "gnn_state.pt")

    mae_gnn, yh_gnn, yt_gnn, ids_gnn = evl(model, make_loader(te_pairs, args.batch_size, False), ym, ysd)
    print(f"\n=== [{args.arch}, matched-subset N={len(df)}] "
          f"GNN holdout (n={len(te_pairs)}) MAE={mae_gnn:.3f} ===", flush=True)

    ens = joblib.load(args.ensemble_path)
    test_df = df[df["_split"] == "test"].set_index(df[df["_split"] == "test"]["id"].astype(str))
    ens_pred = ens.predict(test_df) + test_df[BASELINE_COL].to_numpy()
    ens_pred_by_id = dict(zip(test_df.index, ens_pred))
    gnn_pred_by_id = dict(zip(ids_gnn, yh_gnn))
    true_by_id = dict(zip(ids_gnn, yt_gnn))
    common = [i for i in ids_gnn if i in ens_pred_by_id]
    y_true = np.array([true_by_id[i] for i in common])
    gnn_p = np.array([gnn_pred_by_id[i] for i in common])
    ens_p = np.array([ens_pred_by_id[i] for i in common])
    mae_ens_c = float(np.abs(y_true - ens_p).mean())
    mae_gnn_c = float(np.abs(y_true - gnn_p).mean())
    best_w, best_mae = 0.0, mae_ens_c
    for w in np.arange(0.05, 1.0, 0.05):
        blend = (1 - w) * ens_p + w * gnn_p
        m = float(np.abs(y_true - blend).mean())
        if m < best_mae:
            best_mae, best_w = m, w
    print(f"  ensemble-only={mae_ens_c:.3f} gnn-only={mae_gnn_c:.3f} "
          f"best_blend(w={best_w:.2f})={best_mae:.3f} (delta {best_mae-mae_ens_c:+.3f})", flush=True)

    (out / "models" / "metadata.json").write_text(json.dumps({
        "arch": args.arch, "hidden": args.hidden, "layers": args.layers,
        "n_dist_layers": args.n_dist_layers, "n_heads": args.n_heads, "lr": args.lr,
        "seed": args.seed, "matched_subset_n": len(df),
        "n_train": len(tr_pairs), "n_val": len(va_pairs), "n_test": len(te_pairs),
        "n_params": sum(p.numel() for p in model.parameters()),
        "gnn_holdout_mae": mae_gnn, "ensemble_only_mae": mae_ens_c, "gnn_only_mae": mae_gnn_c,
        "best_blend_w_gnn": float(best_w), "best_blend_mae": float(best_mae),
    }, indent=2))
    print(f"wrote {out}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
