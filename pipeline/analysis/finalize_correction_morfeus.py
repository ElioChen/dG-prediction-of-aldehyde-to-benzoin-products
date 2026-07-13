#!/usr/bin/env python
"""VARIANT of finalize_correction.py: existing 72 champion features + the extra morfeus
descriptors (Pyramidalization + atom-resolved Dispersion P_int, product+aldehyde side),
to test whether they push below the champion's test MAE 1.566/1.540 (see
gxtb-dft-correction-champion). Independent of the RDKit-augmented comparison
(finalize_correction_rdkit.py) -- runs in parallel on the same node.

Same data source, same label file, same 70:20:10 split (seed 42) so the MAE is directly
comparable. Does NOT save a production bundle, only reports the comparison.
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d")

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

PROD_MORF = ["pyr_ketC", "pyr_carbC", "pyr_angle_ketC", "pyr_angle_carbC",
            "disp_p_int_ketC", "disp_p_int_carbC"]
ALD_MORF_RAW = ["pyr_CHO_C", "pyr_angle_CHO_C", "disp_p_int_CHO_C"]
ALD_MORF = [f"ald_{c}" for c in ALD_MORF_RAW]


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
    return [("MLP", MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4, max_iter=250,
                                 early_stopping=True, n_iter_no_change=12)),
            ("XGB_d8", _xgb(8, 1500)), ("XGB_d10", _xgb(10, 2000))]


def run(df, feats, label):
    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    sc = StandardScaler().fit(df[feats].values[tr])
    Xtr, Xva, Xte = sc.transform(df[feats].values[tr]), sc.transform(df[feats].values[va]), sc.transform(df[feats].values[te])
    dtr, dva = df.delta.values[tr], df.delta.values[va]; gte, yte = df.dG_gxtb_kcal.values[te], df.dG_orca_kcal.values[te]

    members = make_members(); preds_te = []
    for nm, m in members:
        if nm.startswith("XGB"): m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
        else: m.fit(Xtr, dtr)
        preds_te.append(m.predict(Xte))
    pred = np.vstack(preds_te).mean(0)
    yhat = gte + pred; err = np.abs(yhat - yte)
    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))
    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
    scope = {}
    for s in ["aromatic", "aliphatic"]:
        mk = df.cls.values[te] == s
        if mk.sum() > 50: scope[s] = float(err[mk].mean())
    print(f"[{label}] n_feat={len(feats)} n={len(df):,} test MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} scope={scope}", flush=True)
    return {"label": label, "n_feat": len(feats), "n": len(df), "mae": mae, "rmse": rmse, "r2": r2, "scope": scope}


def main():
    cons = Path(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet")
    dft = pd.read_parquet(cons, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")

    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id"] + ALD, low_memory=False).drop_duplicates("id").rename(columns={"id": "ald_id", **{c: f"ald_{c}" for c in ALD}})
    cls = pd.read_parquet(f"{H}/aldehyde_class.parquet")

    prod_morf = pd.read_csv(f"{H}/products_morfeus_extra.csv")
    ald_morf = pd.read_csv(f"{H}/aldehydes_morfeus_extra.csv").rename(
        columns={"id": "ald_id", **{c: f"ald_{c}" for c in ALD_MORF_RAW}})

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_morf, on="id", how="left")
    full = full.merge(ald_morf[["ald_id"] + ALD_MORF], on="ald_id", how="left")

    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)

    feats_72 = PROD_QM + ALDp + GLOB
    morf_feats = PROD_MORF + ALD_MORF
    feats_72_plus_morf = feats_72 + morf_feats

    all_feats = sorted(set(feats_72) | set(feats_72_plus_morf))
    df = df.dropna(subset=all_feats + ["dG_gxtb_kcal", "dG_orca_kcal"]).reset_index(drop=True)
    print(f"common labeled rows across all variants: {len(df):,}", flush=True)

    results = {}
    for label, feats in [("baseline_72", feats_72),
                        ("72_plus_morfeus9", feats_72_plus_morf)]:
        results[label] = run(df, feats, label)

    rep = OUT / f"REPORT_morfeus_augment_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# Morfeus-augmented feature comparison ({TAG})\n\n")
        fh.write("Same 70:20:10 split (seed 42), same ensemble (MLP+XGB8+XGB10), vs champion "
                 "gxtb-dft-correction-champion (72 feats, full-100%-label test MAE 1.566).\n\n")
        for label, r in results.items():
            fh.write(f"- **{label}**: n_feat={r['n_feat']} n={r['n']:,} "
                     f"MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} R2={r['r2']:.3f} scope={r['scope']}\n")
    json.dump(results, open(OUT / f"morfeus_augment_results_{TAG}.json", "w"), indent=2)
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
