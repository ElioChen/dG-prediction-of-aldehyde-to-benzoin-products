#!/usr/bin/env python
"""Pure-SMILES (no-xtb) surrogate for product C-C BDFE -- the fast/no-quantum tier, mirroring
this project's established 2D-surrogate pattern (pipeline/train_surrogate.py, which does the
same for the main reaction dG). The QM-feature surrogate (train_bde_surrogate.py) already
gets R2=0.901 for product BDFE using 72-champion QM + mordred feats -- but those need an xtb
geometry+SP to obtain, so they aren't "free" for screening brand-new molecules. This script
tests how much of that predictive power survives using ONLY RDKit 2D descriptors computable
directly from the SMILES string (no conformer, no xtb at all) -- the real no-quantum-cost
comparison point.

Uses the GFN2-level product BDFE (products_bdfe2_descriptors.csv, full library, 96% filled)
as the training target since it's the largest available label set right now.
"""
import sys
import time
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compute"))
from ald_descriptors import calc_rdkit, _RDKIT_FIELDS

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d"); MODELDIR = Path(f"{R}/pipeline/models"); MODELDIR.mkdir(exist_ok=True)


def _xgb(depth, ne):
    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,
                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16,
                        early_stopping_rounds=60, eval_metric="mae")


def savefig(name):
    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", name, flush=True)


def main():
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "smiles"], low_memory=False).drop_duplicates("id")
    bdfe = pd.read_csv(f"{H}/products_bdfe2_descriptors.csv", usecols=["id", "bdfe_xtb_kcal"])
    bdfe = bdfe[bdfe["bdfe_xtb_kcal"].abs() <= 200]  # drop pathological SCF-failure rows

    df = p.merge(bdfe, on="id", how="inner")
    print(f"n={len(df):,} products with valid BDFE label", flush=True)

    print("computing pure-SMILES 2D descriptors (no xtb)...", flush=True)
    t0 = time.time()
    feat_rows = [calc_rdkit(s) for s in df["smiles"]]
    print(f"done in {time.time()-t0:.1f}s", flush=True)
    feat_df = pd.DataFrame(feat_rows)
    df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    feats = _RDKIT_FIELDS
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    d = df.dropna(subset=feats + ["bdfe_xtb_kcal"]).reset_index(drop=True)
    print(f"n={len(d):,} after dropna, n_feat={len(feats)}", flush=True)

    rng = np.random.default_rng(42); idx = rng.permutation(len(d))
    ntr, nva = int(.7 * len(d)), int(.9 * len(d)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    imp = SimpleImputer(strategy="median").fit(d[feats].values[tr])
    Xtr_raw, Xva_raw, Xte_raw = imp.transform(d[feats].values[tr]), imp.transform(d[feats].values[va]), imp.transform(d[feats].values[te])
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)
    ytr, yva, yte = d["bdfe_xtb_kcal"].values[tr], d["bdfe_xtb_kcal"].values[va], d["bdfe_xtb_kcal"].values[te]

    xgb8, xgb10 = _xgb(8, 1500), _xgb(10, 2000)
    xgb8.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    xgb10.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    yhat = np.vstack([xgb8.predict(Xte), xgb10.predict(Xte)]).mean(0)
    err = np.abs(yhat - yte)
    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))
    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
    print(f"[prod_bdfe_smiles2d] n={len(d):,} n_feat={len(feats)} test MAE={mae:.3f} "
         f"RMSE={rmse:.3f} R2={r2:.3f}", flush=True)

    bundle = MODELDIR / f"bde_surrogate_prod_bdfe_smiles2d_{TAG}.joblib"
    joblib.dump({"members": [("XGB_d8", xgb8), ("XGB_d10", xgb10)], "imputer": imp,
                "scaler": sc, "features": feats, "target": "bdfe_xtb_kcal (GFN2)",
                "test_mae": mae, "test_r2": r2, "n_train": len(tr)}, bundle)
    print("wrote", bundle, flush=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, yhat, s=4, alpha=0.25, color="#6a51a3")
    lo, hi = min(yte.min(), yhat.min()), max(yte.max(), yhat.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("real xtb product BDFE (kcal/mol)"); ax.set_ylabel("pure-SMILES surrogate prediction (kcal/mol)")
    ax.set_title(f"product BDFE pure-SMILES(2D) surrogate parity (test, n={len(yte):,})\nMAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    savefig(f"130_parity_prod_bdfe_smiles2d_{TAG}.png")

    imp_gain = xgb8.get_booster().get_score(importance_type="gain")
    top_feats = sorted(imp_gain.items(), key=lambda kv: kv[1], reverse=True)
    feat_names = [feats[int(k[1:])] for k, _ in top_feats]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top_feats))[::-1], [v for _, v in top_feats])
    ax.set_yticks(range(len(top_feats))[::-1]); ax.set_yticklabels(feat_names, fontsize=9)
    ax.set_xlabel("XGB_d8 gain importance")
    ax.set_title("product BDFE pure-SMILES(2D) surrogate feature importance")
    savefig(f"131_feat_importance_prod_bdfe_smiles2d_{TAG}.png")

    rep = OUT / f"REPORT_prod_bdfe_smiles2d_surrogate_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# Product BDFE pure-SMILES (no-xtb) surrogate ({TAG})\n\n")
        fh.write(f"Predicts product C-C BDFE (GFN2, {len(d):,} molecules) from ONLY "
                f"{len(feats)} RDKit 2D descriptors computable directly from the SMILES "
                f"string -- no conformer, no xtb at all. Compare to the QM-feature surrogate "
                f"(`train_bde_surrogate.py`, needs xtb geometry+SP): R2=0.901, MAE=2.70.\n\n")
        fh.write(f"- **pure-SMILES(2D)**: n_feat={len(feats)} n={len(d):,} MAE={mae:.3f} "
                f"RMSE={rmse:.3f} R2={r2:.3f}\n")
        fh.write(f"- Top predictive feats: {', '.join(feat_names[:5])}\n\n")
        gap = 0.901 - r2
        verdict = ("substantial (QM electronic-structure info is doing real work; "
                  "not a cheap substitute)" if gap > 0.15 else
                  "small (2D descriptors capture most of the signal; may be usable as a "
                  "cheap prospective pre-filter)")
        fh.write(f"## Interpretation\n\nR2 gap vs the QM-feature surrogate: {gap:.3f} -- "
                f"**{verdict}**.\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
