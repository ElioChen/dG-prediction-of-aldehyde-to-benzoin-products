#!/usr/bin/env python
"""Phase-1 baseline B3 (BDE_prediction.md section 二/六): "D-SPOC"-style descriptor-
difference + gradient-boosted trees.

x_diff = x(fragment1) + x(fragment2) - x(parent)

using RDKit's built-in physchem Descriptors (no `mordred` dependency -- keeps this in the
already-isolated envs/bde_lite env) as x(.). This is deliberately NOT concatenation (as
plain ECFP/Mordred B2 does): summing the two radical fragments' descriptors and
subtracting the parent's encodes the actual bond-breaking PROCESS (mirrors
BDE = E(fragA) + E(fragB) - E(parent)), which is why the doc calls it a more
"interpretable, reaction-native" baseline than B2 even though both use trees.

Reuses the fragment1/fragment2 SMILES already produced by
pipeline/compute/calc_bde_alfabet.py (ALFABET fragments at exactly the target bond) --
no separate fragmentation step needed. Requires that script's output to already exist.

Usage:
  python train_dspoc_baseline.py --which aldehydes \
      --alfabet-csv /scratch-shared/.../aldehydes_bde_alfabet.csv --out /tmp/dspoc_ald.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from qc import qc_filter
from splits import molecule_cold_split

RDLogger.DisableLog("rdApp.*")
H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")

# A modest, interpretable physchem descriptor set (not the full ~200 RDKit Descriptors --
# keeps each row cheap since this runs 3x per row: parent + 2 fragments).
DESC_FUNCS_SLIM = [
    ("MolWt", Descriptors.MolWt),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("TPSA", Descriptors.TPSA),
    ("MolLogP", Descriptors.MolLogP),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("Kappa2", Descriptors.Kappa2),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
]

# Full RDKit built-in descriptor set (~217, no `mordred` dependency needed to get a
# "richer than 18" feature set). 2026-07-15: added to test whether the aldehyde-side
# D-SPOC underperformance (R2=0.28 vs product-side 0.65 with the SAME slim set) is a
# feature-coverage limitation or a genuine "descriptor-difference doesn't suit this bond"
# effect -- see PROGRESS_20260714.md for the full comparison once both are in.
DESC_FUNCS_FULL = Descriptors._descList

DESC_FUNCS = DESC_FUNCS_SLIM


def descvec(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS)
    except Exception:
        return None
    vec = []
    for _, fn in DESC_FUNCS:
        try:
            v = fn(mol)
        except Exception:
            v = np.nan
        vec.append(v)
    return np.array(vec, dtype=float)


def diff_features(parents, frag1s, frag2s):
    n = len(parents)
    X = np.full((n, len(DESC_FUNCS)), np.nan)
    ok = np.zeros(n, dtype=bool)
    for i, (p, f1, f2) in enumerate(zip(parents, frag1s, frag2s)):
        dp, d1, d2 = descvec(p), descvec(f1), descvec(f2)
        if dp is None or d1 is None or d2 is None:
            continue
        X[i] = d1 + d2 - dp
        ok[i] = True
    return X, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--alfabet-csv", required=True,
                     help="output of calc_bde_alfabet.py (has id, fragment1, fragment2)")
    ap.add_argument("--target", choices=["bde", "bdfe"], default="bde")
    ap.add_argument("--full-descriptors", action="store_true",
                     help="use all ~217 RDKit Descriptors instead of the 18-feature slim set")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global DESC_FUNCS
    if args.full_descriptors:
        DESC_FUNCS = DESC_FUNCS_FULL
    print(f"using {len(DESC_FUNCS)} descriptors", flush=True)

    frags = pd.read_csv(args.alfabet_csv, dtype={"id": str})
    frags = frags.rename(columns={"smiles_canonical": "smiles"}).drop_duplicates("id")

    labels = pd.read_csv(H / f"{args.which}_bdfe_gxtb_descriptors.csv", dtype={"id": str})
    ycol = f"{args.target}_gxtb_kcal"
    labels = labels.dropna(subset=[ycol]).drop_duplicates("id")
    n_before = len(labels)
    labels = labels[qc_filter(labels[ycol])]
    print(f"QC: dropped {n_before - len(labels)}/{n_before} rows outside [20,200] kcal/mol")

    id_cols = ["id", "donor_id"] if args.which == "products" else ["id"]
    extra = None
    if args.which == "products":
        extra = pd.read_csv(H / f"{args.which}_all.csv", usecols=id_cols, dtype=str,
                             keep_default_na=False)

    df = labels.merge(frags, on="id", how="inner")
    if extra is not None:
        df = df.merge(extra, on="id", how="inner")
    print(f"{args.which}: {len(df)} rows with both a BDE label and an ALFABET fragment pair")

    X, ok = diff_features(df["smiles"], df["fragment1"], df["fragment2"])
    df = df[ok].reset_index(drop=True)
    X = X[ok]
    y = df[ycol].to_numpy()

    split_col = "id" if args.which == "aldehydes" else "donor_id"
    split = molecule_cold_split(df[split_col], test_frac=args.test_frac, seed=args.seed)
    tr, te = (split == "train").to_numpy(), (split == "test").to_numpy()
    print(f"train={tr.sum()}  test={te.sum()}  cold on '{split_col}'")

    model = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                          random_state=args.seed, n_jobs=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    mean_pred = np.full(te.sum(), y[tr].mean())

    importances = dict(zip([n for n, _ in DESC_FUNCS], model.feature_importances_.tolist()))
    result = {
        "which": args.which, "target": ycol, "n": len(df),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "model": {
            "MAE": float(mean_absolute_error(y[te], pred)),
            "RMSE": float(root_mean_squared_error(y[te], pred)),
            "R2": float(r2_score(y[te], pred)),
            "spearman_rho": float(spearmanr(y[te], pred).correlation),
        },
        "mean_baseline_MAE": float(mean_absolute_error(y[te], mean_pred)),
        "feature_importance": importances,
    }
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
