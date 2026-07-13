#!/usr/bin/env python3
"""
B-validation: do benzoin PRODUCT descriptors (Multiwfn-free) lower the Δ-model CV MAE?

Merges the product descriptors (pipeline/compute/featurize_product.py output, homo
pairs, funnel_v3, NO Multiwfn) onto the g-xTB-baseline training table and compares,
on the SAME molecules / baseline / QC / CV, three feature tiers — all Multiwfn-free so
the winner is directly feasible at 220k:

  A  aldehyde-only         RDKit-2D + aldehyde QM(no Multiwfn) + g-xTB baseline
  B  aldehyde + product    A + product QM(no Multiwfn)
  (ref) current shipped    full 63 incl Multiwfn (= 2.00) for context

Usage:  python bvalid_product_eval.py [--chunks 'data/raw/product_bvalid/chunks_out/prod_*.csv']
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline")); sys.path.insert(0, str(REPO / "pipeline/compute"))
import delta_core as dc

MWF_PREFIX = ("wfn_", "adch_", "qtaim_")
# product columns that are NOT descriptors (metadata / energies / categoricals)
PROD_DROP = {"index", "donor_smiles", "acceptor_smiles", "product_smiles", "donor_class",
             "acceptor_class", "reaction_type", "is_homo", "error", "xtb_optimized",
             "xyz_file", "product_xyz_file", "dG_xtb_kcal", "G_donor", "G_acceptor",
             "G_product", "G_donor_xtb_Eh", "G_acceptor_xtb_Eh", "G_prod_xtb_Eh",
             "donor_id", "acceptor_id", "idx"}


def _cv(df, feats, baseline_col, target_col, folds=5, repeats=4):
    X = df[feats].copy(); X = X.fillna(X.median(numeric_only=True))
    base = df[baseline_col].to_numpy(float); dft = df[target_col].to_numpy(float)
    y = dft - base
    oof = np.zeros_like(dft); cnt = np.zeros_like(dft)
    for tr, te in RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=42).split(X):
        m = dc.build_model("xgb"); m.fit(X.iloc[tr], y[tr])
        oof[te] += base[te] + m.predict(X.iloc[te]); cnt[te] += 1
    oof /= np.maximum(cnt, 1)
    return dict(n=len(df), nfeat=len(feats),
                mae=mean_absolute_error(dft, oof),
                rmse=float(np.sqrt(mean_squared_error(dft, oof))),
                r2=r2_score(dft, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(REPO / "data/raw/product_bvalid/chunks_out/prod_*.csv"))
    ap.add_argument("--parquet", default=str(REPO / "data/featurize_gxtb_baseline.parquet"))
    args = ap.parse_args()

    # training table (g-xTB baseline in dG_xtb_kcal); apply the same QC/scope as production
    tbl = dc.load_training_table(parquet=args.parquet)        # tbl.df has SMILES, index, feats, target
    base_col, tgt = dc.XTB_DG, tbl.target
    df = tbl.df.copy()

    # product descriptors, keyed by donor_id == training index (homo)
    files = sorted(glob.glob(args.chunks))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            pass  # task still running / empty placeholder
    prod = pd.concat(frames, ignore_index=True)
    # featurize_product writes `index` as a per-CHUNK local row number and drops donor_id,
    # so neither is a usable join key. Merge on CANONICAL donor SMILES instead (homo pairs:
    # donor == acceptor == the aldehyde, which is the training-table SMILES).
    from rdkit import Chem
    def _canon(s):
        m = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
        return Chem.MolToSmiles(m) if m else None
    prod["_canon"] = prod["donor_smiles"].map(_canon)
    pcols = [c for c in prod.columns if c not in PROD_DROP and c != "_canon"
             and not c.startswith(MWF_PREFIX) and pd.api.types.is_numeric_dtype(prod[c])]
    prod = (prod.dropna(subset=["_canon"]).drop_duplicates("_canon")[["_canon"] + pcols]
                .rename(columns={c: f"prod_{c}" for c in pcols}))
    df["_canon"] = df["SMILES"].map(_canon)
    merged = df.merge(prod, on="_canon", how="inner")
    prod_feats = [f"prod_{c}" for c in pcols]
    print(f"product chunks: {len(files)} files, {len(prod)} rows, {len(pcols)} product descriptors")
    print(f"merged training rows (have product desc): {len(merged)}")

    # feature tiers (all Multiwfn-free)
    ald_all = [c for c in tbl.feats if c != base_col]
    ald_nomwf = [c for c in ald_all if not c.startswith(MWF_PREFIX)]
    tiers = {
        "A aldehyde-only (no Mwfn)": ald_nomwf + [base_col],
        "B aldehyde + product (no Mwfn)": ald_nomwf + prod_feats + [base_col],
        "ref full incl Multiwfn (ald)": tbl.feats,
    }
    print(f"\n{'tier':<34} {'n':>5} {'feat':>5} {'MAE':>6} {'RMSE':>6} {'R2':>6}")
    print("-" * 66)
    res = {}
    for name, feats in tiers.items():
        feats = [f for f in feats if f in merged.columns]
        r = _cv(merged, feats, base_col, tgt); res[name] = r
        print(f"{name:<34} {r['n']:>5} {r['nfeat']:>5} {r['mae']:>6.3f} {r['rmse']:>6.3f} {r['r2']:>6.3f}")
    a = res["A aldehyde-only (no Mwfn)"]["mae"]; b = res["B aldehyde + product (no Mwfn)"]["mae"]
    print(f"\nΔMAE adding product descriptors (B − A) = {b - a:+.3f} kcal/mol")
    print("→ product descriptors HELP" if b < a - 0.02 else
          "→ product descriptors do NOT meaningfully help (Multiwfn-free aldehyde model suffices)")


if __name__ == "__main__":
    main()
