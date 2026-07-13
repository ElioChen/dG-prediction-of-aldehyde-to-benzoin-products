#!/usr/bin/env python
"""Full go/no-go for the g-xTB-CONSISTENT BDFE/BDE descriptors (method-mismatch hypothesis,
see bde-descriptor-idea memory): the GFN2-level BDFE was a NULL RESULT
(finalize_correction_bdfe_full.py: +0.007 MAE, noise-level) because the descriptor's level
of theory (GFN2) didn't match the thing it's meant to help correct (g-xTB's own errors).
calc_bde_free_energy_gxtb.py recomputes BOTH bdfe_gxtb_kcal (free-energy-corrected, matches
this project's G_gxtb hybrid-correction pattern) and bde_gxtb_kcal (raw electronic energy,
free byproduct of the same calc -- see bde-descriptor-idea's "also added BDE_gxtb for free"
note) for the aldehyde C(=O)-H bond and the product's new C-C bond, now g-xTB-consistent.

Full-library arrays now complete on both sides (aldehydes job 24460460: 1471/1471, 100%;
products backfill 24454920 + original: 1460/1463, 3 timeout gap accepted) -- this is the
real verdict on the method-mismatch hypothesis, mirroring finalize_correction_bdfe_full.py's
exact protocol (same 70:20:10 split, same quick XGB_d8+XGB_d10 ensemble, same mordredslim271
baseline) but swapping in the g-xTB-consistent sidecars. Tests 4 configs in one run since
both bdfe_gxtb_kcal and bde_gxtb_kcal come from the SAME calc at zero extra compute cost:
  1. mordredslim271                          (baseline, test MAE 1.525 production champion)
  2. + BDFE(gxtb, both sides)                (the pilot's promising aldehyde-side correlation)
  3. + BDE(gxtb, both sides)                 (raw-E; GFN2's raw-E BDE had the only real gain, +0.024)
  4. + BDFE(gxtb) + BDE(gxtb), both sides    (both descriptors, in case they're complementary)
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
FEATS_72 = PROD_QM + ALDp + GLOB


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


def run(df, feats, label):
    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    imp = SimpleImputer(strategy="median").fit(df[feats].values[tr])
    Xtr, Xva, Xte = imp.transform(df[feats].values[tr]), imp.transform(df[feats].values[va]), imp.transform(df[feats].values[te])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    dtr, dva = df.delta.values[tr], df.delta.values[va]; gte, yte = df.dG_gxtb_kcal.values[te], df.dG_orca_kcal.values[te]

    members = [("XGB_d8", _xgb(8, 1500)), ("XGB_d10", _xgb(10, 2000))]
    preds_te = []
    for nm, m in members:
        m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
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

    kept_mordred = set(json.load(open(f"{H}/viz_gxtb_20260625/mordred_slim_selection_20260703.json"))["kept_mordred"])
    prod_kept = [c for c in kept_mordred if not c.startswith("ald_")]
    ald_kept_raw = [c[len("ald_"):] for c in kept_mordred if c.startswith("ald_")]

    prod_header = pd.read_csv(f"{H}/products_mordred_descriptors.csv", nrows=0).columns
    prod_want = ["id"] + [c for c in prod_header if c in prod_kept]
    prod_mrd = pd.read_csv(f"{H}/products_mordred_descriptors.csv", usecols=prod_want, low_memory=False)
    ald_header = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", nrows=0).columns
    ald_want = ["id"] + [c for c in ald_header if c in ald_kept_raw]
    ald_mrd = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", usecols=ald_want, low_memory=False)
    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={"id": "ald_id"})
    ald_mrd_raw = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={c: f"ald_{c}" for c in ald_mrd_raw})
    ald_mrd_cols = [f"ald_{c}" for c in ald_mrd_raw]
    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")

    # g-xTB-CONSISTENT BDFE/BDE sidecars (both sides), freshly concatenated from the now-
    # complete arrays (aldehydes 1471/1471, products 1460/1463).
    prod_bdfe = pd.read_csv(f"{H}/products_bdfe_gxtb_descriptors.csv",
                             usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"bdfe_gxtb_kcal": "prod_bdfe_gxtb_kcal", "bde_gxtb_kcal": "prod_bde_gxtb_kcal"})
    ald_bdfe = pd.read_csv(f"{H}/aldehydes_bdfe_gxtb_descriptors.csv",
                            usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"id": "ald_id", "bdfe_gxtb_kcal": "ald_bdfe_gxtb_kcal", "bde_gxtb_kcal": "ald_bde_gxtb_kcal"})
    # sanity filter: drop pathological non-converged-SCF rows (|value|>200 kcal/mol impossible
    # for a real bond), same threshold as the GFN2 version.
    for c in ["prod_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal"]:
        prod_bdfe.loc[prod_bdfe[c].abs() > 200, c] = np.nan
    for c in ["ald_bdfe_gxtb_kcal", "ald_bde_gxtb_kcal"]:
        ald_bdfe.loc[ald_bdfe[c].abs() > 200, c] = np.nan

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")
    full = full.merge(prod_bdfe, on="id", how="left")
    full = full.merge(ald_bdfe, on="ald_id", how="left")

    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS_72).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print(f"labeled rows: {len(df):,}", flush=True)
    print(f"g-xTB BDFE coverage: prod {df['prod_bdfe_gxtb_kcal'].notna().mean()*100:.1f}%  "
         f"ald {df['ald_bdfe_gxtb_kcal'].notna().mean()*100:.1f}%", flush=True)
    print(f"g-xTB BDE coverage:  prod {df['prod_bde_gxtb_kcal'].notna().mean()*100:.1f}%  "
         f"ald {df['ald_bde_gxtb_kcal'].notna().mean()*100:.1f}%", flush=True)

    feats_slim271 = FEATS_72 + prod_mrd_cols + ald_mrd_cols
    feats_bdfe = feats_slim271 + ["prod_bdfe_gxtb_kcal", "ald_bdfe_gxtb_kcal"]
    feats_bde = feats_slim271 + ["prod_bde_gxtb_kcal", "ald_bde_gxtb_kcal"]
    feats_both = feats_slim271 + ["prod_bdfe_gxtb_kcal", "ald_bdfe_gxtb_kcal",
                                   "prod_bde_gxtb_kcal", "ald_bde_gxtb_kcal"]
    print(f"mordredslim271: {len(feats_slim271)} feats; "
          f"+BDFE(gxtb): {len(feats_bdfe)}; +BDE(gxtb): {len(feats_bde)}; +both: {len(feats_both)}", flush=True)

    results = {}
    configs = [
        ("mordredslim271", feats_slim271),
        ("mordredslim271_plus_bdfe_gxtb_both", feats_bdfe),
        ("mordredslim271_plus_bde_gxtb_both", feats_bde),
        ("mordredslim271_plus_bdfe_and_bde_gxtb_both", feats_both),
    ]
    for label, feats in configs:
        results[label] = run(df, feats, label)

    rep = OUT / f"REPORT_bdfe_gxtb_full_augment_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# g-xTB-consistent BDFE/BDE augmented feature comparison ({TAG})\n\n")
        fh.write("Method-mismatch hypothesis test: GFN2-level BDFE was a null result "
                 "(finalize_correction_bdfe_full.py, +0.007 MAE). This uses g-xTB-consistent "
                 "BDFE/BDE (calc_bde_free_energy_gxtb.py), full library both sides "
                 "(aldehydes 1471/1471, products 1460/1463, 3 timeout gap accepted). "
                 "Same 70:20:10 split (seed 42), XGB_d8+XGB_d10 ensemble, vs mordredslim271 "
                 "preferred production model (test MAE 1.525).\n\n")
        for label, r in results.items():
            fh.write(f"- **{label}**: n_feat={r['n_feat']} n={r['n']:,} "
                     f"MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} R2={r['r2']:.3f} scope={r['scope']}\n")
    json.dump(results, open(OUT / f"bdfe_gxtb_full_augment_results_{TAG}.json", "w"), indent=2)
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
