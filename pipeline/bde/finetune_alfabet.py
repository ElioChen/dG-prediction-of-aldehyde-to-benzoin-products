#!/usr/bin/env python
"""Phase-1 baseline B1 (BDE_prediction.md section 六): fine-tune ALFABET's pretrained
weights on this project's own g-xTB BDE labels, rather than just using it zero-shot (B0).

Scope note: this fine-tunes ONLY the final linear readout for the "bde" head
(`bde_no_mean`, a single Dense(128->1) layer, 129 params -- see the model summary: the
message-passing body (~1.67M params across 6 edge/node-update blocks) stays FROZEN).
Reasons this is the right scope rather than full end-to-end retraining:
  1. The zero-shot diagnosis (calc_bde_alfabet.py pilots) found ALFABET's raw aldehyde
     formyl C-H predictions cluster very tightly (85-94 kcal/mol, consistent with real
     literature values ~88-89) -- the message-passing body has already learned sensible
     bond-environment chemistry; what's missing is calibration to THIS project's own
     g-xTB BDE scale/sensitivity, which a linear readout retrain can capture without
     touching (and risking destabilizing) the pretrained graph representation.
  2. ALFABET's own per-bond output is a padded (batch, max_bonds, 1) tensor gathered by
     bond_index; batching requires left-padding bond/atom arrays with a phantom entry
     (get_features(pad=True)) whose exact offset interaction with bond_index was not
     worth reverse-engineering under time pressure -- this script instead does plain
     batch-of-1 forward/backward passes (no padding ambiguity at all, matching
     alfabet.prediction.tf_model_forward's own single-molecule convention) and
     accumulates gradients over `--accum` molecules before each optimizer step.

Reuses calc_bde_alfabet.py's target_bond_aldehyde/target_bond_product (already fixed for
the canonical-vs-original atom-order bug) so the bond_index used for fine-tuning is
guaranteed consistent with the zero-shot bond identification.

Usage:
  ENV=/gpfs/scratch1/shared/schen3/envs/alfabet
  $ENV/bin/python finetune_alfabet.py --which aldehydes --n 5000 --epochs 3 \
      --out /tmp/alfabet_finetune_ald.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compute"))
from qc import qc_filter  # noqa: E402
from splits import molecule_cold_split  # noqa: E402

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--target", choices=["bde", "bdfe"], default="bde")
    ap.add_argument("--n", type=int, default=5000, help="subsample size (fine-tuning "
                     "only 129 params, doesn't need the full ~220k library)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=16, help="molecules per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from calc_bde_alfabet import target_bond_aldehyde, target_bond_product
    finder = target_bond_aldehyde if args.which == "aldehydes" else target_bond_product

    labels = pd.read_csv(H / f"{args.which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    ycol = f"{args.target}_gxtb_kcal"
    labels = labels.dropna(subset=[ycol]).drop_duplicates("id")
    labels = labels[qc_filter(labels[ycol])]

    id_cols = ["id", "smiles"] if args.which == "aldehydes" else ["id", "donor_id", "smiles"]
    mol = pd.read_csv(H / f"{args.which}_all.csv", usecols=id_cols, dtype=str,
                       keep_default_na=False)
    df = labels.merge(mol, on="id", how="inner")
    df = df[df["smiles"] != ""].sample(n=min(args.n, len(df)), random_state=args.seed) \
        .reset_index(drop=True)

    split_col = "id" if args.which == "aldehydes" else "donor_id"
    split = molecule_cold_split(df[split_col], test_frac=args.test_frac, seed=args.seed)

    targets, bidx, ok = [], [], np.zeros(len(df), dtype=bool)
    for i, smi in enumerate(df["smiles"]):
        hit = finder(smi)
        if hit is None:
            continue
        csmi, b = hit
        targets.append(csmi)
        bidx.append(b)
        ok[i] = True
    df = df[ok].reset_index(drop=True)
    split = split[ok.tolist() if isinstance(ok, list) else ok].reset_index(drop=True) \
        if hasattr(split, "reset_index") else split[ok]
    y = df[ycol].to_numpy(dtype=float)
    print(f"{args.which}: {len(df)}/{args.n} matched target bond", flush=True)

    tr_idx = np.where((split == "train").to_numpy())[0]
    te_idx = np.where((split == "test").to_numpy())[0]
    print(f"train={len(tr_idx)}  test={len(te_idx)}  cold on '{split_col}'", flush=True)

    import tensorflow as tf
    from alfabet.preprocessor import get_features
    from alfabet.prediction import model as alfabet_model

    for layer in alfabet_model.layers:
        layer.trainable = (layer.name == "bde_no_mean")
    optim = tf.keras.optimizers.Adam(learning_rate=args.lr)
    trainable_vars = [w for layer in alfabet_model.layers if layer.trainable
                       for w in layer.trainable_weights]
    print(f"fine-tuning {sum(np.prod(v.shape) for v in trainable_vars)} params "
          f"({[v.name for v in trainable_vars]})", flush=True)

    def forward_bde(i):
        feats = get_features(targets[i])
        inputs = {k: tf.constant(np.expand_dims(v, 0)) for k, v in feats.items()}
        bde_pred, _ = alfabet_model(inputs, training=True)
        return tf.squeeze(bde_pred)[bidx[i]]

    def evaluate(idx):
        preds = []
        for i in idx:
            feats = get_features(targets[i])
            inputs = {k: tf.constant(np.expand_dims(v, 0)) for k, v in feats.items()}
            bde_pred, _ = alfabet_model(inputs, training=False)
            preds.append(float(tf.squeeze(bde_pred)[bidx[i]].numpy()))
        return np.array(preds)

    pred0 = evaluate(te_idx)  # zero-shot baseline, for a same-script before/after comparison
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    def metrics(y_true, y_pred):
        return dict(MAE=float(mean_absolute_error(y_true, y_pred)),
                    R2=float(r2_score(y_true, y_pred)),
                    spearman_rho=float(spearmanr(y_true, y_pred).correlation))

    zero_shot = metrics(y[te_idx], pred0)
    print("zero-shot (this script's own eval):", zero_shot, flush=True)

    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = rng.permutation(tr_idx)
        epoch_loss, n_steps = 0.0, 0
        for start in range(0, len(order), args.accum):
            chunk = order[start:start + args.accum]
            with tf.GradientTape() as tape:
                preds = tf.stack([forward_bde(i) for i in chunk])
                y_true = tf.constant(y[chunk], dtype=preds.dtype)
                loss = tf.reduce_mean(tf.square(preds - y_true))
            grads = tape.gradient(loss, trainable_vars)
            optim.apply_gradients(zip(grads, trainable_vars))
            epoch_loss += float(loss); n_steps += 1
        print(f"epoch {epoch}: mean_batch_mse={epoch_loss / max(n_steps,1):.4f}", flush=True)

    pred1 = evaluate(te_idx)
    fine_tuned = metrics(y[te_idx], pred1)
    print("fine-tuned:", fine_tuned, flush=True)

    result = {
        "which": args.which, "target": ycol, "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)), "zero_shot": zero_shot, "fine_tuned": fine_tuned,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
