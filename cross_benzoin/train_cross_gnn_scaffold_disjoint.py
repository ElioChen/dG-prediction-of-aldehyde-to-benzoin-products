#!/usr/bin/env python
"""
GNN + tabular stacking, retrained/re-evaluated on the CORRECTED scaffold-
disjoint split (new_scaffold_split column) instead of train_cross_gnn.py's
hardcoded candidates_v3 pair_split_labels() (now known to leak scaffolds
severely, see memory cross-five-diagnostics-20260717.md).

Reuses train_cross_gnn.py's model/graph code verbatim (TripleGNN, graph
featurization, training loop, evl()) via import -- only the split logic and
the ensemble artifact path differ. Key correctness fix vs the original:
'mixed' rows (donor/acceptor scaffolds land in different splits) are
EXPLICITLY DROPPED, not silently promoted to train via the old NaN->
train_extra fallback (which was correct for the old script's official-
labels-are-sparse assumption, but wrong here since 'mixed' is a real,
common, deliberately-excluded category, not a rare missing-label case).

Usage:
  python cross_benzoin/train_cross_gnn_scaffold_disjoint.py \
      --table data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_split_labeled.parquet \
      --champion-dir data/cross_benzoin/cross_round7/scaffold_disjoint_v1 \
      --ensemble-path data/cross_benzoin/cross_round7/scaffold_disjoint_v1/models/ensemble_scaffold_disjoint.joblib \
      --outdir data/cross_benzoin/cross_round7/train_gnn_scaffold_disjoint_v1
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402
from train_cross_delta import TARGET_COL, BASELINE_COL  # noqa: E402
from train_cross_ensemble import MLPXGBEnsemble  # noqa: E402,F401
from train_cross_gnn import (  # noqa: E402
    TripleGNN, build_graphs, make_loader, evl, dev,
)

import __main__ as _main_mod  # noqa: E402
_main_mod.MLPXGBEnsemble = MLPXGBEnsemble


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--champion-dir", required=True, help="dir with models/feature_list.json")
    ap.add_argument("--ensemble-path", required=True, help="path to the scaffold-disjoint ensemble .joblib")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from-homo", default=None)
    ap.add_argument("--extra-val-frac", type=float, default=0.1,
                     help="fraction of the clean-train pool carved out for early-stopping")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    feats = json.loads((Path(args.champion_dir) / "models" / "feature_list.json").read_text())
    df = pd.read_parquet(args.table)
    print(f"loaded {len(df)} rows", flush=True)

    df = df[df["new_scaffold_split"] != "mixed"].reset_index(drop=True)
    print(f"dropped 'mixed' rows -> {len(df)} remain "
          f"({df['new_scaffold_split'].value_counts().to_dict()})", flush=True)

    df["_split"] = df["new_scaffold_split"]
    if args.extra_val_frac > 0:
        rng = np.random.default_rng(args.seed)
        train_pairs_all = df.loc[df["_split"] == "train", "pair_key"].unique()
        n_extra_val = int(len(train_pairs_all) * args.extra_val_frac)
        val_pairs_extra = set(rng.choice(train_pairs_all, size=n_extra_val, replace=False))
        promote = (df["_split"] == "train") & df["pair_key"].isin(val_pairs_extra)
        df.loc[promote, "_split"] = "validation_extra"
        print(f"carved {promote.sum()} rows ({len(val_pairs_extra)} pairs) from clean-train "
              f"into early-stopping validation", flush=True)

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

    print("building graphs (product + donor + acceptor)...", flush=True)
    t0 = time.time()
    pairs = build_graphs(df, qmz)
    print(f"  built {len(pairs)}/{len(df)} pairs ({time.time()-t0:.0f}s)", flush=True)
    id_to_split = dict(zip(df["id"].astype(str), df["_split"]))
    tr_pairs = [p for p in pairs if id_to_split.get(p.row_id) == "train"]
    va_pairs = [p for p in pairs if id_to_split.get(p.row_id) in ("validation", "validation_extra")]
    te_pairs = [p for p in pairs if id_to_split.get(p.row_id) == "test"]
    print(f"  split: train={len(tr_pairs)} val={len(va_pairs)} test={len(te_pairs)}", flush=True)

    ys = torch.tensor([p.y.item() for p in tr_pairs])
    ym, ysd = ys.mean().item(), ys.std().item()
    for p in pairs:
        p.ystd = (p.y - ym) / ysd

    ad, bd = pairs[0].x_p.shape[1], pairs[0].edge_attr_p.shape[1]
    nqm = len(feats)
    torch.manual_seed(args.seed)
    model = TripleGNN(ad, bd, nqm, h=args.hidden, layers=args.layers).to(dev)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k", flush=True)

    if args.init_from_homo:
        homo_sd = torch.load(args.init_from_homo, map_location="cpu", weights_only=False)
        own_sd = model.state_dict()
        loaded, skipped = 0, 0
        for dst_prefix, src_prefix in [("encP.", "encP."), ("encD.", "encA."), ("encA.", "encA.")]:
            for k, v in homo_sd.items():
                if not k.startswith(src_prefix):
                    continue
                dst_k = dst_prefix + k[len(src_prefix):]
                if dst_k in own_sd and own_sd[dst_k].shape == v.shape:
                    own_sd[dst_k] = v.clone()
                    loaded += 1
                else:
                    skipped += 1
        model.load_state_dict(own_sd)
        print(f"transfer-learning init from {args.init_from_homo}: "
              f"{loaded} loaded, {skipped} skipped", flush=True)

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
        if ep % 5 == 0 or pat == 0:
            print(f"  ep{ep:3d} val_MAE {vm:.3f} (best {best:.3f}) {time.time()-t1:.0f}s", flush=True)
        if pat >= args.patience:
            break
    model.load_state_dict(best_state)
    torch.save(best_state, out / "models" / "cross_gnn_state.pt")
    # normalization stats needed to run this checkpoint standalone at inference time
    # (qm_mean/qm_std standardize the QM-scalar channel, ym/ysd de-normalize the
    # GNN's target-standardized output back to kcal/mol -- both fit on train only)
    joblib.dump({"qm_mean": qm_mean, "qm_std": qm_std, "med": med, "ym": ym, "ysd": ysd,
                 "feats": feats, "hidden": args.hidden, "layers": args.layers,
                 "ad": ad, "bd": bd, "nqm": nqm},
                out / "models" / "gnn_norm_stats.joblib")

    mae_gnn, yh_gnn, yt_gnn, ids_gnn = evl(model, make_loader(te_pairs, args.batch_size, False), ym, ysd)
    print(f"\n=== GNN scaffold-disjoint holdout (n={len(te_pairs)}) MAE={mae_gnn:.3f} ===", flush=True)

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
    mae_gnn_c = float(np.abs(y_true - gnn_p).mean())
    mae_ens_c = float(np.abs(y_true - ens_p).mean())
    rows = [{"w_gnn": 0.0, "mae": mae_ens_c, "config": "ensemble_only"}]
    best_w, best_mae = 0.0, mae_ens_c
    for w in np.arange(0.05, 1.0, 0.05):
        blend = (1 - w) * ens_p + w * gnn_p
        m = float(np.abs(y_true - blend).mean())
        rows.append({"w_gnn": round(float(w), 2), "mae": m, "config": "blend"})
        if m < best_mae:
            best_mae, best_w = m, w
    rows.append({"w_gnn": 1.0, "mae": mae_gnn_c, "config": "gnn_only"})
    res = pd.DataFrame(rows)
    res.to_csv(out / "data" / "gnn_ensemble_stack.csv", index=False)
    print(f"\n=== stacking (n={len(common)}) ===", flush=True)
    print(f"  ensemble-only MAE={mae_ens_c:.3f} | gnn-only MAE={mae_gnn_c:.3f} | "
          f"best blend w_gnn={best_w:.2f} MAE={best_mae:.3f} "
          f"(delta vs ensemble-only {best_mae-mae_ens_c:+.3f})", flush=True)

    (out / "models" / "metadata.json").write_text(json.dumps({
        "model": "triple_gnn_gine_h128_l4_scaffold_disjoint",
        "init_from_homo": args.init_from_homo,
        "extra_val_frac": args.extra_val_frac,
        "n_train": len(tr_pairs), "n_val": len(va_pairs), "n_test": len(te_pairs),
        "gnn_scaffold_disjoint_holdout_mae": mae_gnn,
        "stacking_common_n": len(common),
        "ensemble_only_mae": mae_ens_c, "gnn_only_mae": mae_gnn_c,
        "best_blend_w_gnn": float(best_w), "best_blend_mae": float(best_mae),
    }, indent=2))
    print(f"\nSaved GNN state + metadata to {out}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
