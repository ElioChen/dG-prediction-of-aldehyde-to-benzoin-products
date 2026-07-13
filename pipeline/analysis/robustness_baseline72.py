#!/usr/bin/env python
"""Overfitting / robustness / reproducibility check for baseline_72.

Two modes, one JSON result file per invocation (aggregated later by
collect_robustness_baseline72.py):

  --mode holdout --seed N   : same 70:20:10 RATIO but a DIFFERENT random permutation seed
                              (tests: is the reported MAE a lucky split, or stable across
                              reshuffles?). Reports train/val/test MAE/RMSE/R2 each time,
                              which also answers the overfitting question directly
                              (train << test would mean overfitting).
  --mode cv --fold N --nfolds K : proper K-fold cross-validation. Each fold's training
                              portion is further split into an inner train/val (90/10) for
                              early stopping; the held-out fold is the test set. Reports the
                              same train/val/test triple per fold.

Point ensemble only (MLP + XGB_d8 + XGB_d10, matching baseline_72's champion point
prediction) -- the quantile/uncertainty-routing models are orthogonal to this question and
skipped to keep the N-way sweep affordable.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625/robustness"); OUT.mkdir(exist_ok=True, parents=True)

PROD_QM = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega",
  "xtb_dipole","mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH",
  "wbo_CO_ket","wbo_CC_new","wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC",
  "fukui_plus_carbC","fukui_minus_carbC","dual_carbC","vbur_ketC","vbur_carbC","sterimol_L",
  "sterimol_B1","sterimol_B5","SASA_total","P_int","pa_ketO","hb_dist","hb_angle","dih_core"]
ALD = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
  "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C",
  "wbo_CO","pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
ALDp = [f"ald_{c}" for c in ALD]
GKEYS = ["TPSA","HBD","HBA","RotB","FracCsp3","nHetero","MolWt","nRing","nAromRing","nAliphRing",
         "nAmide","has_P","has_B","has_S","has_Si","has_halogen"]
GLOB = [f"g_{k}" for k in GKEYS]
FEATS = PROD_QM + ALDp + GLOB


def gfeats(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return {f"g_{k}": np.nan for k in GKEYS}
    s = {a.GetSymbol() for a in m.GetAtoms()}
    vals = [rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),
            rdMolDescriptors.CalcNumHeteroatoms(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumAliphaticRings(m),
            rdMolDescriptors.CalcNumAmideBonds(m), int('P' in s), int('B' in s), int('S' in s),
            int('Si' in s), int(bool(s & {'F','Cl','Br','I'}))]
    return {f"g_{k}": v for k, v in zip(GKEYS, vals)}


def add_global(df, smi_col):
    u = df[[smi_col]].drop_duplicates()
    g = pd.DataFrame([gfeats(s) for s in u[smi_col]]); g[smi_col] = u[smi_col].values
    return df.merge(g, on=smi_col, how="left")


def load_data():
    cons = Path(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet")
    dft = pd.read_parquet(cons, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id"] + ALD, low_memory=False).drop_duplicates("id").rename(columns={"id": "ald_id", **{c: f"ald_{c}" for c in ALD}})
    cls = pd.read_parquet(f"{H}/aldehyde_class.parquet")
    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a, on="ald_id", how="left")
    full = add_global(full, "smiles")
    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    return df


def fit_eval(df, tr, va, te, label):
    sc = StandardScaler().fit(df[FEATS].values[tr])
    Xtr, Xva, Xte = sc.transform(df[FEATS].values[tr]), sc.transform(df[FEATS].values[va]), sc.transform(df[FEATS].values[te])
    dtr, dva = df.delta.values[tr], df.delta.values[va]

    mlp = MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4, max_iter=250,
                       early_stopping=True, n_iter_no_change=12)
    mlp.fit(Xtr, dtr)

    def xgb(depth, ne):
        return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,
                            colsample_bytree=0.7, min_child_weight=5, n_jobs=16,
                            early_stopping_rounds=60, eval_metric="mae")
    xgb8, xgb10 = xgb(8, 1500), xgb(10, 2000)
    xgb8.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
    xgb10.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
    members = [mlp, xgb8, xgb10]

    def metrics(split_idx, Xs):
        g, y = df.dG_gxtb_kcal.values[split_idx], df.dG_orca_kcal.values[split_idx]
        pred = np.vstack([m.predict(Xs) for m in members]).mean(0)
        yhat = g + pred; err = np.abs(yhat - y)
        mae = float(err.mean()); rmse = float(np.sqrt(((yhat - y) ** 2).mean()))
        r2 = float(1 - ((yhat - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(split_idx))}

    result = {"label": label,
             "train": metrics(tr, Xtr), "val": metrics(va, Xva), "test": metrics(te, Xte)}
    print(f"[{label}] train MAE={result['train']['mae']:.3f}  val MAE={result['val']['mae']:.3f}  "
         f"test MAE={result['test']['mae']:.3f}  gap(test-train)={result['test']['mae']-result['train']['mae']:.3f}",
         flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["holdout", "cv"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--nfolds", type=int, default=5)
    args = ap.parse_args()

    df = load_data()
    print(f"labeled (72-feat) {len(df):,}", flush=True)

    if args.mode == "holdout":
        rng = np.random.default_rng(args.seed); idx = rng.permutation(len(df))
        ntr, nva = int(.7 * len(df)), int(.9 * len(df))
        tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
        label = f"holdout_seed{args.seed}"
        result = fit_eval(df, tr, va, te, label)
        result["mode"] = "holdout"; result["seed"] = args.seed
        outf = OUT / f"result_holdout_seed{args.seed}.json"
    else:
        kf = KFold(n_splits=args.nfolds, shuffle=True, random_state=123)
        splits = list(kf.split(np.arange(len(df))))
        train_full, te = splits[args.fold]
        # inner 90/10 split of the fold's training portion for early-stopping val
        rng = np.random.default_rng(1000 + args.fold)
        perm = rng.permutation(len(train_full))
        ninner = int(.9 * len(train_full))
        tr, va = train_full[perm[:ninner]], train_full[perm[ninner:]]
        label = f"cv_fold{args.fold}of{args.nfolds}"
        result = fit_eval(df, tr, va, te, label)
        result["mode"] = "cv"; result["fold"] = args.fold; result["nfolds"] = args.nfolds
        outf = OUT / f"result_cv_fold{args.fold}.json"

    json.dump(result, open(outf, "w"), indent=2)
    print("wrote", outf, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
