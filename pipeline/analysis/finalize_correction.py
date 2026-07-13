#!/usr/bin/env python
"""FINALIZE the g-xTB->DFT correction on the (now-complete) DFT labels.

Production model = Δ-learning (DFT - g-xTB) on **72 features**:
  56 QM (34 product + 22 reactant/aldehyde) + 16 RDKit global (whole-molecule) descriptors.
ENSEMBLE = MLP + 3 XGB(seeds) -> prediction = mean, **uncertainty = std across members**.
UNCERTAINTY ROUTING: molecules with std above the threshold are flagged route_to_dft
(don't trust the correction) -- targets the P/B/S / large-flexible tail (see no-dG-extreme-filtering).

Saves model bundle + per-molecule full-library output (corrected ΔG + uncertainty + route flag).
Runs on CPU (sklearn + xgboost + rdkit). Re-run any time more DFT labels land (dated tag).
"""
import glob, json, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d"); MODELDIR = Path(f"{R}/pipeline/models"); MODELDIR.mkdir(exist_ok=True)
ROUTE_FRAC = 0.15   # flag the most-uncertain 15% as route-to-DFT

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
FEATS = PROD_QM + ALDp + GLOB   # 72


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


def _xgb(depth, ne):
    return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,
                        colsample_bytree=0.7, min_child_weight=5, n_jobs=16,
                        early_stopping_rounds=60, eval_metric="mae")

def make_members():
    # point-prediction ensemble: MLP + XGB depth-8 + XGB depth-10 (deeper XGB best single, exp5)
    return [("MLP", MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4, max_iter=250,
                                 early_stopping=True, n_iter_no_change=12)),
            ("XGB_d8", _xgb(8, 1500)), ("XGB_d10", _xgb(10, 2000))]

def make_quantiles():
    # quantile regressors for a calibrated prediction interval -> routing signal (exp3: better separation)
    return {q: XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, n_estimators=800,
                            max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, n_jobs=16)
            for q in (0.05, 0.95)}


def main():
    # prefer the consolidated single-source label file; fall back to globbing chunk CSVs
    cons = Path(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet")
    if cons.exists():
        dft = pd.read_parquet(cons, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")
    else:
        fs = sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/chunk_*.csv")) + \
             sorted(glob.glob(f"{R}/data/raw/dft_sp_funnelv3/retry7200/chunk_*.csv"))
        dft = pd.concat([pd.read_csv(f, usecols=["id", "dG_orca_kcal"]) for f in fs], ignore_index=True)
        dft = dft.dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id"] + ALD, low_memory=False).drop_duplicates("id").rename(columns={"id": "ald_id", **{c: f"ald_{c}" for c in ALD}})
    cls = pd.read_parquet(f"{H}/aldehyde_class.parquet")
    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a, on="ald_id", how="left")
    full = add_global(full, "smiles")
    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print(f"labeled (72-feat) {len(df):,}  DFT coverage {len(dft):,}/219421", flush=True)

    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    sc = StandardScaler().fit(df[FEATS].values[tr])
    Xtr, Xva, Xte = sc.transform(df[FEATS].values[tr]), sc.transform(df[FEATS].values[va]), sc.transform(df[FEATS].values[te])
    dtr, dva = df.delta.values[tr], df.delta.values[va]; gte, yte = df.dG_gxtb_kcal.values[te], df.dG_orca_kcal.values[te]

    members = make_members(); preds_te = []
    for nm, m in members:
        if nm.startswith("XGB"): m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
        else: m.fit(Xtr, dtr)
        preds_te.append(m.predict(Xte))
    pred = np.vstack(preds_te).mean(0)               # point = ensemble mean
    # uncertainty = quantile prediction-interval width (better routing separation than ensemble-std, exp3)
    quant = make_quantiles()
    for q, m in quant.items(): m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
    unc = quant[0.95].predict(Xte) - quant[0.05].predict(Xte)
    yhat = gte + pred; err = np.abs(yhat - yte)
    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))
    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
    thr = float(np.quantile(unc, 1 - ROUTE_FRAC))
    conf = unc < thr
    conf = unc < thr
    print(f"ENSEMBLE(72; MLP+XGB8+XGB10) test MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}", flush=True)
    print(f"quantile-PI routing thr(width)={thr:.3f}  confident {conf.mean()*100:.0f}%  conf-MAE={err[conf].mean():.3f}  routed-MAE={err[~conf].mean():.3f}  sep={err[~conf].mean()/err[conf].mean():.2f}x", flush=True)
    scope = {}
    for s in ["aromatic", "aliphatic"]:
        mk = df.cls.values[te] == s
        if mk.sum() > 50: scope[s] = {"MAE": float(err[mk].mean()), "conf_MAE": float(err[mk & conf].mean())}

    # save bundle (point members + quantile models + scaler + features + routing threshold)
    bundle = MODELDIR / f"gxtb_dft_correction_ENSEMBLE72_{TAG}.joblib"
    joblib.dump({"members": members, "quantiles": quant, "scaler": sc, "features": FEATS, "global_keys": GKEYS,
                 "route_width_threshold": thr, "route_frac": ROUTE_FRAC,
                 "target": "DFT_r2scan3c - gxtb (kcal/mol)", "test_mae": mae, "n_train": len(tr)}, bundle)
    json.dump({"model": "ensemble_MLP+XGB8+XGB10 / quantile-PI routing", "n_feat": len(FEATS),
               "test_mae": mae, "test_rmse": rmse, "test_r2": r2, "route_width_threshold": thr,
               "confident_frac": float(conf.mean()), "confident_mae": float(err[conf].mean()),
               "routed_mae": float(err[~conf].mean()), "scope": scope, "n_labeled": len(df), "dft_coverage": int(len(dft))},
              open(MODELDIR / f"gxtb_dft_correction_{TAG}.json", "w"), indent=2)

    # full-library prediction + uncertainty + route flag
    fl = full.dropna(subset=["dG_gxtb_kcal"] + FEATS).copy()
    Xf = sc.transform(fl[FEATS].values)
    fl["delta_pred"] = np.vstack([m.predict(Xf) for _, m in members]).mean(0)
    fl["uncertainty_pi_width"] = quant[0.95].predict(Xf) - quant[0.05].predict(Xf)
    fl["dG_gxtb_corrected_final"] = fl["dG_gxtb_kcal"] + fl["delta_pred"]
    fl["route_to_dft"] = fl["uncertainty_pi_width"] >= thr
    fl[["id", "smiles", "dG_gxtb_kcal", "dG_gxtb_corrected_final", "uncertainty_pi_width", "route_to_dft"]].to_csv(
        OUT / f"products_dG_corrected_FINAL_{TAG}.csv", index=False)

    rep = OUT / f"REPORT_FINAL_correction_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# FINAL g-xTB->DFT correction ({TAG})\n\n")
        fh.write(f"- DFT labels: **{len(dft):,}/219,421** ({100*len(dft)/219421:.0f}%); training set {len(df):,}\n")
        fh.write(f"- Model: **ensemble MLP+3×XGB on 72 feats** (56 QM + 16 RDKit global), Δ-learning\n")
        fh.write(f"- **TEST MAE {mae:.3f}, RMSE {rmse:.3f}, R² {r2:.3f}**\n")
        fh.write(f"- Uncertainty routing: flag most-uncertain {ROUTE_FRAC*100:.0f}% (std≥{thr:.2f}) -> route to DFT\n")
        fh.write(f"  - confident {conf.mean()*100:.0f}% MAE **{err[conf].mean():.3f}** | routed MAE {err[~conf].mean():.3f}\n")
        fh.write(f"- Scope: {scope}\n")
        fh.write(f"- Model: `{bundle}`\n- Full-library output: `products_dG_corrected_FINAL_{TAG}.csv` "
                 f"({len(fl):,} mols; {int(fl.route_to_dft.sum()):,} flagged route_to_dft)\n")
    print("wrote", rep, "and", bundle, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
