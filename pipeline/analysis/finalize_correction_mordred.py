#!/usr/bin/env python
"""VARIANT of finalize_correction.py: existing 72 champion features + a TARGETED subset of
Mordred descriptors, to test whether the SHAP-motivated dispersion/size-shape angle pushes
below the champion floor (baseline_72 ~1.54-1.58, see descriptor-search-exhausted).

NB: the full 1826-descriptor dump (3325 usable cols x2 after coverage filtering) was tried
first and TIMED OUT TWICE at 3h (jobs 24385045 with MLP, 24390274 XGB-only) -- XGB training
on ~3300 dims doesn't finish in a reasonable time even without the MLP. Also this build of
mordred has NO WHIM/GETAWAY/RDF modules (checked via Calculator descriptor module names) --
the actual dispersion/size/shape-relevant families available are: MoRSE (160, 3D electron-
diffraction-based), CPSA (43, charged partial surface area), Polarizability (2, direct
dispersion proxy), GeometricalIndex/MomentOfInertia/PBF/McGowanVolume/VdwVolumeABC (shape+
volume), Weight/TopoPSA (size). ~219/side, 438 total -- an order of magnitude smaller than
the full dump, trains in reasonable time, and is the actually-targeted test of the SHAP
hypothesis (matches the original proposal before "run everything" was chosen).

Products array had 4/1099 chunks (~800 mols) time out and are simply missing -- NOT
backfilled, per instruction to proceed with whatever's done rather than block.

Same 70:20:10 split (seed 42) as finalize_correction.py so MAE is directly comparable. Does
NOT save a production bundle, only reports the comparison (quick go/no-go, matches the
RDKit/morfeus precedent -- escalate to full training-runs-full-diagnostics treatment only if
this shows a real signal).
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')

# dispersion/size/shape-relevant mordred modules (see module docstring for why these and
# not WHIM/GETAWAY/RDF, which this mordred build doesn't have)
TARGET_MODULES = {"MoRSE", "CPSA", "Polarizability", "GeometricalIndex", "MomentOfInertia",
                  "PBF", "McGowanVolume", "VdwVolumeABC", "Weight", "TopoPSA"}


def targeted_mordred_names() -> set[str]:
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=False)
    return {str(d) for d in calc.descriptors if type(d).__module__.split(".")[-1] in TARGET_MODULES}

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
    # back to the full 3-member ensemble now that the targeted subset (~510 total dims,
    # similar width to the RDKit-434 comparison which trained fine) replaced the full
    # 3300+-dim dump that made MLP/XGB training intractable.
    return [("MLP", MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4, max_iter=250,
                                 early_stopping=True, n_iter_no_change=12)),
            ("XGB_d8", _xgb(8, 1500)), ("XGB_d10", _xgb(10, 2000))]


def run(df, feats, label):
    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    # median-impute (fit on train only, avoids leakage) before scaling -- mordred's ~1782
    # columns each have a small independent NaN rate, so almost every ROW has >=1 NaN
    # somewhere even though coverage per-column is high; row-dropping wiped out the whole
    # dataset (see 2026-07-02 failed run), so impute instead of drop.
    imp = SimpleImputer(strategy="median").fit(df[feats].values[tr])
    Xtr_raw, Xva_raw, Xte_raw = imp.transform(df[feats].values[tr]), imp.transform(df[feats].values[va]), imp.transform(df[feats].values[te])
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)
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

    target_names = targeted_mordred_names()
    print(f"targeted dispersion/size/shape mordred names: {len(target_names)}", flush=True)

    print("loading mordred sidecars (targeted subset only)...", flush=True)
    prod_header = pd.read_csv(f"{H}/products_mordred_descriptors.csv", nrows=0).columns
    prod_want = ["id"] + [c for c in prod_header if c.replace("mordred_", "") in target_names]
    prod_mrd = pd.read_csv(f"{H}/products_mordred_descriptors.csv", usecols=prod_want, low_memory=False)
    ald_header = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", nrows=0).columns
    ald_want = ["id"] + [c for c in ald_header if c.replace("mordred_", "") in target_names]
    ald_mrd = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", usecols=ald_want, low_memory=False)

    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={"id": "ald_id"})
    ald_mrd_cols_raw = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={c: f"ald_{c}" for c in ald_mrd_cols_raw})
    ald_mrd_cols = [f"ald_{c}" for c in ald_mrd_cols_raw]
    # mordred can emit non-numeric error sentinels in some cells; coerce to numeric
    for c in prod_mrd_cols:
        prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    for c in ald_mrd_cols:
        ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")
    # keep columns with >=80% coverage (drop near-universally-failing descriptors entirely);
    # remaining scattered NaNs are median-imputed inside run(), not dropped.
    prod_mrd_cols = [c for c in prod_mrd_cols if prod_mrd[c].notna().mean() >= 0.80]
    ald_mrd_cols = [c for c in ald_mrd_cols if ald_mrd[c].notna().mean() >= 0.80]
    print(f"mordred usable cols (targeted + >=80% coverage): product {len(prod_mrd_cols)}, "
         f"aldehyde {len(ald_mrd_cols)}", flush=True)

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")

    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)

    feats_72 = PROD_QM + ALDp + GLOB
    mordred_feats = prod_mrd_cols + ald_mrd_cols
    feats_72_plus_mordred = feats_72 + mordred_feats

    # only require the champion's own 72 feats + labels complete (as finalize_correction.py
    # does) -- mordred NaNs are median-imputed in run(), not used to filter rows, or the
    # ~1782-wide scattered-NaN pattern would drop ~all rows (see note above).
    df = df.dropna(subset=feats_72 + ["dG_gxtb_kcal", "dG_orca_kcal"]).reset_index(drop=True)
    print(f"common labeled rows (72-feat complete): {len(df):,}", flush=True)

    results = {}
    for label, feats in [("baseline_72", feats_72),
                        (f"72_plus_mordred{len(mordred_feats)}", feats_72_plus_mordred)]:
        results[label] = run(df, feats, label)

    rep = OUT / f"REPORT_mordred_augment_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# Mordred-augmented feature comparison ({TAG})\n\n")
        fh.write("Same 70:20:10 split (seed 42), same ensemble (MLP+XGB8+XGB10), vs baseline_72 "
                 "noise band 1.571+/-0.013 (see REPORT_robustness_baseline72_20260702.md). "
                 f"4/1099 product chunks (~800 mols, ~0.36%) timed out and are simply missing "
                 "(dropna), not backfilled, per instruction to proceed with what's done.\n\n")
        for label, r in results.items():
            fh.write(f"- **{label}**: n_feat={r['n_feat']} n={r['n']:,} "
                     f"MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} R2={r['r2']:.3f} scope={r['scope']}\n")
    json.dump(results, open(OUT / f"mordred_augment_results_{TAG}.json", "w"), indent=2)
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
