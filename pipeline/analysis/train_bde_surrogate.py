#!/usr/bin/env python
"""BDE/BDFE surrogate model: predict per-molecule bond dissociation (free) energy directly
from the cheap descriptors already computed for every molecule (72-champion QM feats + the
mordredslim271 kept mordred subset), instead of always running real xtb (--ohess for BDFE).
See bde-surrogate-model-idea memory for the two motivations:
  1. Redundancy diagnostic -- if a nonlinear model reconstructs BDE/BDFE well from existing
     feats (high R2), it's mostly redundant info; if it reconstructs poorly, it's genuinely
     new information the existing feats don't capture (consistent with the earlier
     correlation analysis showing max |r|~0.5 with any single feature).
  2. Prospective-scoring speed -- a good surrogate gives near-instant BDE/BDFE for NEW
     aldehyde/product candidates outside the current 220k library, without running xtb.

Trains 4 independent single-molecule regressors (aldehyde raw-E BDE, aldehyde BDFE, product
raw-E BDE, product BDFE), each XGB_d8+XGB_d10 ensemble + quantile(05/95) UQ, same 70:20:10
split (seed 42) as the rest of this project. Full diagnostics per target per
training-runs-full-diagnostics memory.
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d"); MODELDIR = Path(f"{R}/pipeline/models"); MODELDIR.mkdir(exist_ok=True)

PROD_QM = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega",
  "xtb_dipole","mulliken_ketC","mulliken_ketO","mulliken_carbC","mulliken_hydO","mulliken_hydH",
  "wbo_CO_ket","wbo_CC_new","wbo_CO_carb","fukui_plus_ketC","fukui_minus_ketC","dual_ketC",
  "fukui_plus_carbC","fukui_minus_carbC","dual_carbC","vbur_ketC","vbur_carbC","sterimol_L",
  "sterimol_B1","sterimol_B5","SASA_total","P_int","pa_ketO","hb_dist","hb_angle","dih_core"]
ALD_QM = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
  "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C",
  "wbo_CO","pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]
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


def make_quantiles():
    return {q: XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, n_estimators=800,
                            max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, n_jobs=16)
            for q in (0.05, 0.95)}


def savefig(name):
    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", name, flush=True)


def train_one(df, feats, target, tag, side_label):
    """Train XGB_d8+XGB_d10 ensemble + quantile UQ for a single BDE/BDFE target. Returns dict
    of results and writes bundle + full diagnostics."""
    d = df.dropna(subset=feats + [target]).reset_index(drop=True)
    rng = np.random.default_rng(42); idx = rng.permutation(len(d))
    ntr, nva = int(.7 * len(d)), int(.9 * len(d)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    imp = SimpleImputer(strategy="median").fit(d[feats].values[tr])
    Xtr_raw, Xva_raw, Xte_raw = imp.transform(d[feats].values[tr]), imp.transform(d[feats].values[va]), imp.transform(d[feats].values[te])
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)
    ytr, yva, yte = d[target].values[tr], d[target].values[va], d[target].values[te]

    xgb8, xgb10 = _xgb(8, 1500), _xgb(10, 2000)
    xgb8.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    xgb10.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    members = [("XGB_d8", xgb8), ("XGB_d10", xgb10)]
    yhat = np.vstack([m.predict(Xte) for _, m in members]).mean(0)
    err = np.abs(yhat - yte)
    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))
    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
    print(f"[{tag}] n={len(d):,} n_feat={len(feats)} test MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}", flush=True)

    quant = make_quantiles()
    for q, m in quant.items(): m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    unc = quant[0.95].predict(Xte) - quant[0.05].predict(Xte)

    bundle = MODELDIR / f"bde_surrogate_{tag}_{TAG}.joblib"
    joblib.dump({"members": members, "quantiles": quant, "imputer": imp, "scaler": sc,
                "features": feats, "target": target, "test_mae": mae, "test_r2": r2,
                "n_train": len(tr)}, bundle)
    print("wrote", bundle, flush=True)

    # ── diagnostics ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, yhat, s=4, alpha=0.25, color="#2171b5")
    lo, hi = min(yte.min(), yhat.min()), max(yte.max(), yhat.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"real xtb {target} (kcal/mol)"); ax.set_ylabel("surrogate prediction (kcal/mol)")
    ax.set_title(f"{side_label} surrogate parity (test, n={len(yte):,})\nMAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    savefig(f"110_parity_{tag}_{TAG}.png")

    resid = yhat - yte
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(resid, bins=80, color="#6baed6", edgecolor="none")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("surrogate - real (kcal/mol)"); ax.set_ylabel("count")
    ax.set_title(f"{side_label} surrogate residual distribution\nmean={resid.mean():.3f} std={resid.std():.3f}")
    savefig(f"111_residual_hist_{tag}_{TAG}.png")

    for m, name, idx_fig in [(xgb8, "XGB_d8", "112"), (xgb10, "XGB_d10", "113")]:
        ev = m.evals_result()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ev["validation_0"]["mae"], label="train MAE"); ax.plot(ev["validation_1"]["mae"], label="val MAE")
        ax.axvline(m.best_iteration, color="k", ls="--", lw=1, label=f"best_iter={m.best_iteration}")
        ax.set_xlabel("boosting round"); ax.set_ylabel("MAE (kcal/mol)")
        ax.set_title(f"{side_label} {name} training curve"); ax.legend()
        savefig(f"{idx_fig}_{name.lower()}_curve_{tag}_{TAG}.png")

    imp_gain = xgb8.get_booster().get_score(importance_type="gain")
    top_feats = sorted(imp_gain.items(), key=lambda kv: kv[1], reverse=True)[:20]
    feat_names = [feats[int(k[1:])] for k, _ in top_feats]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(top_feats))[::-1], [v for _, v in top_feats])
    ax.set_yticks(range(len(top_feats))[::-1]); ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_xlabel("XGB_d8 gain importance")
    ax.set_title(f"{side_label} surrogate top-20 feature importance")
    savefig(f"114_feat_importance_{tag}_{TAG}.png")

    return {"tag": tag, "target": target, "n": len(d), "n_feat": len(feats), "mae": mae,
            "rmse": rmse, "r2": r2, "bundle": str(bundle), "top_feats": feat_names[:10]}


def main():
    kept_mordred = set(json.load(open(f"{H}/viz_gxtb_20260625/mordred_slim_selection_20260703.json"))["kept_mordred"])
    prod_kept = {c for c in kept_mordred if not c.startswith("ald_")}
    ald_kept = {c[len("ald_"):] for c in kept_mordred if c.startswith("ald_")}

    # ── aldehyde side ────────────────────────────────────────────
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id", "smiles"] + ALD_QM, low_memory=False).drop_duplicates("id")
    a = add_global(a, "smiles")
    ald_header = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", nrows=0).columns
    ald_want = ["id"] + [c for c in ald_header if c in ald_kept]
    ald_mrd = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", usecols=ald_want, low_memory=False)
    ald_mrd_cols = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")
    a = a.merge(ald_mrd, on="id", how="left")

    ald_bde = pd.read_csv(f"{H}/aldehydes_bde_descriptors.csv", usecols=["id", "bde_ald_CH_kcal"])
    ald_bdfe = pd.read_csv(f"{H}/aldehydes_bdfe2_descriptors.csv", usecols=["id", "bdfe_xtb_kcal"])
    ald_bdfe = ald_bdfe[ald_bdfe["bdfe_xtb_kcal"].abs() <= 200]  # drop pathological SCF-failure rows

    a = a.merge(ald_bde, on="id", how="left").merge(ald_bdfe, on="id", how="left")
    ald_feats = ALD_QM + GLOB + ald_mrd_cols
    print(f"aldehyde: n={len(a):,}, feats={len(ald_feats)}", flush=True)
    print(f"  BDE(E) coverage: {a['bde_ald_CH_kcal'].notna().mean()*100:.1f}%  "
         f"BDFE coverage: {a['bdfe_xtb_kcal'].notna().mean()*100:.1f}%", flush=True)

    # ── product side ─────────────────────────────────────────────
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "smiles"] + PROD_QM, low_memory=False).drop_duplicates("id")
    p = add_global(p, "smiles")
    prod_header = pd.read_csv(f"{H}/products_mordred_descriptors.csv", nrows=0).columns
    prod_want = ["id"] + [c for c in prod_header if c in prod_kept]
    prod_mrd = pd.read_csv(f"{H}/products_mordred_descriptors.csv", usecols=prod_want, low_memory=False)
    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith("mordred_")]
    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    p = p.merge(prod_mrd, on="id", how="left")

    prod_bde = pd.read_csv(f"{H}/products_bde_descriptors.csv", usecols=["id", "bde_prod_CC_kcal"])
    prod_bdfe = pd.read_csv(f"{H}/products_bdfe2_descriptors.csv", usecols=["id", "bdfe_xtb_kcal"])
    prod_bdfe = prod_bdfe[prod_bdfe["bdfe_xtb_kcal"].abs() <= 200]

    p = p.merge(prod_bde, on="id", how="left").merge(prod_bdfe, on="id", how="left")
    prod_feats = PROD_QM + GLOB + prod_mrd_cols
    print(f"product: n={len(p):,}, feats={len(prod_feats)}", flush=True)
    print(f"  BDE(E) coverage: {p['bde_prod_CC_kcal'].notna().mean()*100:.1f}%  "
         f"BDFE coverage: {p['bdfe_xtb_kcal'].notna().mean()*100:.1f}%", flush=True)

    results = {}
    results["ald_bde"] = train_one(a, ald_feats, "bde_ald_CH_kcal", "ald_bde", "aldehyde BDE(raw-E)")
    results["ald_bdfe"] = train_one(a, ald_feats, "bdfe_xtb_kcal", "ald_bdfe", "aldehyde BDFE")
    results["prod_bde"] = train_one(p, prod_feats, "bde_prod_CC_kcal", "prod_bde", "product BDE(raw-E)")
    results["prod_bdfe"] = train_one(p, prod_feats, "bdfe_xtb_kcal", "prod_bdfe", "product BDFE")

    rep = OUT / f"REPORT_bde_surrogate_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# BDE/BDFE surrogate model ({TAG})\n\n")
        fh.write("Predicts per-molecule bond dissociation (free) energy from the existing cheap "
                 "descriptors (72-champion QM + mordredslim271 kept mordred subset), instead of "
                 "running real xtb (--ohess for BDFE). Same 70:20:10 split (seed 42), XGB_d8+"
                 "XGB_d10 ensemble + quantile(05/95) UQ, per target.\n\n")
        fh.write("## Results\n\n")
        fh.write("| target | n | n_feat | test MAE | RMSE | R2 |\n|---|---|---|---|---|---|\n")
        for tag, r in results.items():
            fh.write(f"| {r['target']} ({tag}) | {r['n']:,} | {r['n_feat']} | {r['mae']:.3f} | "
                     f"{r['rmse']:.3f} | {r['r2']:.3f} |\n")
        fh.write("\n## Interpretation\n\n")
        for tag, r in results.items():
            redund = "HIGH (mostly redundant with existing feats)" if r["r2"] > 0.85 else \
                     ("MODERATE" if r["r2"] > 0.6 else "LOW (carries genuinely independent information)")
            fh.write(f"- **{tag}**: R2={r['r2']:.3f} -> redundancy is **{redund}**. "
                     f"Top predictive feats: {', '.join(r['top_feats'][:5])}\n")
        fh.write("\nIf R2 is high enough for practical use (rule of thumb R2>0.85, MAE well under "
                 "the real xtb run's own noise), the corresponding bundle can be used as a fast "
                 "prospective-screening substitute for real xtb BDE/BDFE on new molecules "
                 "outside the current 220k library.\n")
        fh.write("\n## Model bundles\n\n")
        for tag, r in results.items():
            fh.write(f"- `{r['bundle']}`\n")
    json.dump(results, open(OUT / f"bde_surrogate_results_{TAG}.json", "w"), indent=2)
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
