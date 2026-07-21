#!/usr/bin/env python
"""Phase-1 baseline B2 (BDE_prediction.md, section 六): ECFP + gradient-boosted trees,
predicting the project's own g-xTB BDE (aldehyde formyl C-H / product ketC-carbC) directly
from the 2D SMILES. Purpose is NOT to replace the existing dG_gxtb-correction pipeline --
it's a fast, generic 2D structure->BDE model to compare against the ALFABET zero-shot
baseline (pipeline/compute/calc_bde_alfabet.py) and, later, feed a homo-trained BDE model
into whatever needs it for cross products (BDE_prediction.md Phase 1: "训练 ECFP/Mordred...
D-SPOC 和 bond-centered D-MPNN").

Uses XGBoost (not literally LightGBM) to match this project's existing model-building
convention (delta_core.build_model) rather than adding a new boosting-library dependency.

Split is molecule-level cold start (pipeline/bde/splits.py:molecule_cold_split) keyed on
the SOURCE ALDEHYDE id -- for products this means the whole homo-product derived from a
given aldehyde moves together, so no aldehyde's chemistry leaks between train and test via
its own product. This is the single-molecule analogue of the donor/acceptor double
cold-start split (same module) used for real cross pairs.

Usage:
  python train_bde_baseline.py --which aldehydes --out /tmp/bde_ecfp_ald.json
  python train_bde_baseline.py --which products  --out /tmp/bde_ecfp_prod.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from qc import qc_filter
from splits import molecule_cold_split

RDLogger.DisableLog("rdApp.*")
H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")


def ecfp(smiles_list, n_bits=2048, radius=2):
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    ok = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bit = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(bit, arr)
        fps[i] = arr
        ok[i] = True
    return fps, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--target", choices=["bde", "bdfe"], default="bde")
    ap.add_argument("--source-id-col", default=None,
                     help="molecule id to key the cold split on; defaults to 'id' for "
                          "aldehydes and 'donor_id' for products (homo => donor==acceptor "
                          "==source aldehyde)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels = pd.read_csv(H / f"{args.which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    ycol = f"{args.target}_gxtb_kcal"
    labels = labels.dropna(subset=[ycol]).drop_duplicates("id")
    # A handful of rows have wildly divergent SCF energies (e.g. 1.1e6 kcal/mol) that
    # were never NaN-filtered upstream -- physical-plausibility + MAD QC (qc.py) before
    # fitting, or a single row dominates MSE/MAE entirely.
    n_before = len(labels)
    labels = labels[qc_filter(labels[ycol])]
    print(f"QC: dropped {n_before - len(labels)}/{n_before} rows (qc.qc_filter)")

    id_cols = ["id", "smiles"] if args.which == "aldehydes" else ["id", "donor_id", "smiles"]
    mol = pd.read_csv(H / f"{args.which}_all.csv", usecols=id_cols, dtype=str,
                       keep_default_na=False)
    df = labels.merge(mol, on="id", how="inner")
    df = df[df["smiles"] != ""].reset_index(drop=True)

    split_col = args.source_id_col or ("id" if args.which == "aldehydes" else "donor_id")
    print(f"{args.which}: {len(df)} labeled rows, cold-splitting on '{split_col}' "
          f"({df[split_col].nunique()} unique molecules)")

    X, ok = ecfp(df["smiles"].tolist())
    df = df[ok].reset_index(drop=True)
    X = X[ok]
    y = df[ycol].to_numpy()

    split = molecule_cold_split(df[split_col], test_frac=args.test_frac, seed=args.seed)
    tr, te = (split == "train").to_numpy(), (split == "test").to_numpy()
    print(f"train={tr.sum()}  test={te.sum()}")

    model = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                          random_state=args.seed, n_jobs=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])

    mean_pred = np.full(te.sum(), y[tr].mean())
    result = {
        "which": args.which, "target": ycol, "split_col": split_col,
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "model": {
            "MAE": float(mean_absolute_error(y[te], pred)),
            "RMSE": float(root_mean_squared_error(y[te], pred)),
            "R2": float(r2_score(y[te], pred)),
            "spearman_rho": float(spearmanr(y[te], pred).correlation),
        },
        "mean_baseline": {
            "MAE": float(mean_absolute_error(y[te], mean_pred)),
            "RMSE": float(root_mean_squared_error(y[te], mean_pred)),
        },
    }
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
