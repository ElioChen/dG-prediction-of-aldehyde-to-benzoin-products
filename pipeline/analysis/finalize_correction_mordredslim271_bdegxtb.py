#!/usr/bin/env python
"""FULL PRODUCTION run of the new champion candidate: MORDREDSLIM271 (72 champion QM +
199 SHAP-pruned mordred, test MAE 1.525) + the 4 g-xTB-consistent BDE/BDFE features
(ald_bde_gxtb_kcal, ald_bdfe_gxtb_kcal, prod_bde_gxtb_kcal, prod_bdfe_gxtb_kcal) that beat
the noise band on the quick 2-member XGB check (1.612 -> 1.563, see
finalize_correction_bdfe_gxtb_full.py / REPORT_bdfe_gxtb_full_augment_20260706.md).

This mirrors finalize_correction_mordred_slim.py's exact production treatment (MLP + 2xXGB
ensemble + quantile-UQ uncertainty routing + full diagnostics) instead of that quick-check's
bare 2-member XGB comparison, to get the real production-grade MAE number on 275 features.
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d"); MODELDIR = Path(f"{R}/pipeline/models"); MODELDIR.mkdir(exist_ok=True)
ROUTE_FRAC = 0.15
NAME = "MORDREDSLIM271_BDEGXTB"

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
BDE_COLS = ["prod_bdfe_gxtb_kcal", "ald_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal", "ald_bde_gxtb_kcal"]


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


def main():
    cons = Path(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet")
    dft = pd.read_parquet(cons, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")

    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id", "smiles"] + ALD, low_memory=False).drop_duplicates("id")
    a_r = a.rename(columns={"id": "ald_id", "smiles": "ald_smiles", **{c: f"ald_{c}" for c in ALD}})
    cls = pd.read_parquet(f"{H}/aldehyde_class.parquet")

    # SHAP-pruned mordredslim271 selection (job 24405069, REPORT_mordred_slim_20260703.md)
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
    ald_mrd_cols_raw = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={c: f"ald_{c}" for c in ald_mrd_cols_raw})
    ald_mrd_cols = [f"ald_{c}" for c in ald_mrd_cols_raw]
    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")
    print(f"slim mordred: {len(prod_mrd_cols)+len(ald_mrd_cols)}/199 kept cols found in sidecars", flush=True)

    # g-xTB-consistent BDE/BDFE sidecars (both sides), the 4 new candidate features
    prod_bde = pd.read_csv(f"{H}/products_bdfe_gxtb_descriptors.csv",
                            usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"bdfe_gxtb_kcal": "prod_bdfe_gxtb_kcal", "bde_gxtb_kcal": "prod_bde_gxtb_kcal"})
    ald_bde = pd.read_csv(f"{H}/aldehydes_bdfe_gxtb_descriptors.csv",
                           usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"id": "ald_id", "bdfe_gxtb_kcal": "ald_bdfe_gxtb_kcal", "bde_gxtb_kcal": "ald_bde_gxtb_kcal"})
    # sanity filter: non-converged-SCF garbage (|value|>200 kcal/mol impossible for a real bond)
    for c in ["prod_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal"]:
        prod_bde.loc[prod_bde[c].abs() > 200, c] = np.nan
    for c in ["ald_bdfe_gxtb_kcal", "ald_bde_gxtb_kcal"]:
        ald_bde.loc[ald_bde[c].abs() > 200, c] = np.nan

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a_r, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")
    full = full.merge(prod_bde, on="id", how="left")
    full = full.merge(ald_bde, on="ald_id", how="left")

    FEATS = FEATS_72 + prod_mrd_cols + ald_mrd_cols + BDE_COLS
    print(f"total features: {len(FEATS)} (72 champion + {len(prod_mrd_cols)+len(ald_mrd_cols)} mordred + {len(BDE_COLS)} g-xTB BDE/BDFE)", flush=True)

    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    # only require the core 72 + labels complete; mordred + BDE/BDFE sparse blocks are
    # median-imputed (fit on train only) rather than dropped, per descriptor-search-exhausted's
    # footgun lesson (union-of-NaN over ~1800+ sparse cols would leave ~0 rows otherwise)
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS_72).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print(f"labeled rows: {len(df):,}", flush=True)
    print(f"g-xTB BDE/BDFE coverage: prod_bde {df['prod_bde_gxtb_kcal'].notna().mean()*100:.1f}% "
          f"ald_bde {df['ald_bde_gxtb_kcal'].notna().mean()*100:.1f}% "
          f"prod_bdfe {df['prod_bdfe_gxtb_kcal'].notna().mean()*100:.1f}% "
          f"ald_bdfe {df['ald_bdfe_gxtb_kcal'].notna().mean()*100:.1f}%", flush=True)

    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    imp = SimpleImputer(strategy="median").fit(df[FEATS].values[tr])
    Xtr_raw, Xva_raw, Xte_raw = imp.transform(df[FEATS].values[tr]), imp.transform(df[FEATS].values[va]), imp.transform(df[FEATS].values[te])
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)
    dtr, dva = df.delta.values[tr], df.delta.values[va]
    gte, yte = df.dG_gxtb_kcal.values[te], df.dG_orca_kcal.values[te]

    # ── train members, keeping training-history curves ──────────────────────
    mlp = MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4, max_iter=250,
                       early_stopping=True, n_iter_no_change=12, validation_fraction=0.1)
    mlp.fit(Xtr, dtr)
    xgb8, xgb10 = _xgb(8, 1500), _xgb(10, 2000)
    xgb8.fit(Xtr, dtr, eval_set=[(Xtr, dtr), (Xva, dva)], verbose=False)
    xgb10.fit(Xtr, dtr, eval_set=[(Xtr, dtr), (Xva, dva)], verbose=False)
    members = [("MLP", mlp), ("XGB_d8", xgb8), ("XGB_d10", xgb10)]

    preds_te = [m.predict(Xte) for _, m in members]
    pred = np.vstack(preds_te).mean(0)

    quant = make_quantiles()
    for q, m in quant.items(): m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
    unc = quant[0.95].predict(Xte) - quant[0.05].predict(Xte)

    yhat = gte + pred; err = np.abs(yhat - yte)
    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))
    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
    thr = float(np.quantile(unc, 1 - ROUTE_FRAC))
    conf = unc < thr
    print(f"{NAME} test MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}", flush=True)
    print(f"confident {conf.mean()*100:.0f}% MAE={err[conf].mean():.3f}  routed MAE={err[~conf].mean():.3f}", flush=True)

    # ── save production bundle ───────────────────────────────────────────
    bundle = MODELDIR / f"gxtb_dft_correction_{NAME}_{TAG}.joblib"
    joblib.dump({"members": members, "quantiles": quant, "imputer": imp, "scaler": sc,
                "features": FEATS, "route_width_threshold": thr, "route_frac": ROUTE_FRAC,
                "target": "DFT_r2scan3c - gxtb (kcal/mol)", "test_mae": mae, "n_train": len(tr)}, bundle)
    print("wrote", bundle, flush=True)

    # ── full-library prediction ──────────────────────────────────────────
    fl = full.dropna(subset=["dG_gxtb_kcal"] + FEATS_72).copy()
    Xf = sc.transform(imp.transform(fl[FEATS].values))
    fl["delta_pred"] = np.vstack([m.predict(Xf) for _, m in members]).mean(0)
    fl["uncertainty_pi_width"] = quant[0.95].predict(Xf) - quant[0.05].predict(Xf)
    fl["dG_gxtb_corrected_final"] = fl["dG_gxtb_kcal"] + fl["delta_pred"]
    fl["route_to_dft"] = fl["uncertainty_pi_width"] >= thr
    fl[["id", "smiles", "dG_gxtb_kcal", "dG_gxtb_corrected_final", "uncertainty_pi_width", "route_to_dft"]].to_csv(
        OUT / f"products_dG_corrected_{NAME}_{TAG}.csv", index=False)
    print(f"wrote full-library predictions ({len(fl):,} mols)", flush=True)

    # ══ diagnostics (same shape as finalize_correction_mordred_slim.py, 100-series to avoid
    # clobbering the 90-99 mordredslim271 series) ═══════════════════════════
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, yhat, s=4, alpha=0.25, color="#2171b5")
    lo, hi = min(yte.min(), yhat.min()), max(yte.max(), yhat.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("DFT dG_orca (kcal/mol)"); ax.set_ylabel("corrected prediction (kcal/mol)")
    ax.set_title(f"{NAME} parity (test, n={len(yte):,})\nMAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    savefig(f"100_parity_{NAME}.png")

    resid = yhat - yte
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(resid, bins=80, color="#6baed6", edgecolor="none")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("prediction - DFT (kcal/mol)"); ax.set_ylabel("count")
    ax.set_title(f"{NAME} residual distribution\nmean={resid.mean():.3f} std={resid.std():.3f}")
    savefig(f"101_residual_hist_{NAME}.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(yhat, resid, s=4, alpha=0.25, color="#cb181d")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("predicted dG (kcal/mol)"); ax.set_ylabel("residual (pred - DFT)")
    ax.set_title(f"{NAME} residual vs predicted")
    savefig(f"102_residual_vs_pred_{NAME}.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mlp.loss_curve_, label="train loss")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        ax2 = ax.twinx(); ax2.plot(mlp.validation_scores_, color="#cb181d", label="val R2")
        ax2.set_ylabel("validation R2", color="#cb181d")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title(f"{NAME} MLP training curve (stopped at epoch {len(mlp.loss_curve_)})")
    savefig(f"103_mlp_loss_curve_{NAME}.png")

    for m, name, fname in [(xgb8, "XGB_d8", f"104_xgb_d8_curve_{NAME}.png"),
                           (xgb10, "XGB_d10", f"105_xgb_d10_curve_{NAME}.png")]:
        ev = m.evals_result()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ev["validation_0"]["mae"], label="train MAE"); ax.plot(ev["validation_1"]["mae"], label="val MAE")
        ax.axvline(m.best_iteration, color="k", ls="--", lw=1, label=f"best_iter={m.best_iteration}")
        ax.set_xlabel("boosting round"); ax.set_ylabel("MAE (kcal/mol)")
        ax.set_title(f"{NAME} {name} training curve"); ax.legend()
        savefig(fname)

    err_thr = float(np.quantile(err, 1 - ROUTE_FRAC))
    y_true_bin = (err >= err_thr).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, unc); roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2171b5", lw=2, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"{NAME} uncertainty-routing ROC\n(worst {ROUTE_FRAC*100:.0f}% by true error)")
    ax.legend(loc="lower right")
    savefig(f"106_uncertainty_roc_{NAME}.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(unc[conf], err[conf], s=4, alpha=0.3, color="#2171b5", label=f"confident ({conf.mean()*100:.0f}%)")
    ax.scatter(unc[~conf], err[~conf], s=4, alpha=0.3, color="#cb181d", label=f"routed ({(~conf).mean()*100:.0f}%)")
    ax.axvline(thr, color="k", ls="--", lw=1)
    ax.set_xlabel("quantile PI width (uncertainty)"); ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title(f"{NAME} error vs uncertainty (conf MAE={err[conf].mean():.3f}, routed MAE={err[~conf].mean():.3f})")
    ax.legend()
    savefig(f"107_error_vs_uncertainty_{NAME}.png")

    molwt = df["g_MolWt"].values[te]; scope_te = df["cls"].values[te]
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, c in [("aromatic", "#2171b5"), ("aliphatic", "#cb181d")]:
        mk = scope_te == s
        ax.scatter(molwt[mk], err[mk], s=4, alpha=0.3, color=c, label=s)
    ax.set_xlabel("product MolWt (g/mol)"); ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title(f"{NAME} error vs molecular size"); ax.legend()
    savefig(f"108_error_vs_molwt_{NAME}.png")

    te_df = df.iloc[te].reset_index(drop=True).copy()
    te_df["dG_pred"] = yhat; te_df["error"] = err; te_df["uncertainty_pi_width"] = unc
    te_df["route_to_dft"] = ~conf
    keep = ["id", "ald_id", "smiles", "ald_smiles", "cls", "dG_orca_kcal", "dG_gxtb_kcal",
            "dG_pred", "error", "uncertainty_pi_width", "route_to_dft"] + BDE_COLS
    te_df[keep].to_csv(OUT / f"test_predictions_{NAME}_{TAG}.csv", index=False)

    worst = te_df.sort_values("error", ascending=False).head(50).reset_index(drop=True)
    worst[keep].to_csv(OUT / f"worst_mispredictions_{NAME}_{TAG}.csv", index=False)
    top20 = worst.head(20)
    mols, legends = [], []
    for _, r in top20.iterrows():
        m = Chem.MolFromSmiles(str(r["ald_smiles"]))
        if m is None: continue
        mols.append(m)
        legends.append(f"id={r['id']} err={r['error']:.2f}\npred={r['dG_pred']:.1f} true={r['dG_orca_kcal']:.1f}")
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(260, 220), legends=legends)
        img.save(OUT / f"109_worst_mispredicted_aldehydes_{NAME}.png")
        print(f"wrote 109_worst_mispredicted_aldehydes_{NAME}.png", flush=True)

    rep = OUT / f"REPORT_{NAME}_FINAL_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# {NAME} production model ({TAG})\n\n")
        fh.write(f"- Features: **{len(FEATS)}** (mordredslim271's 271 [72 champion QM + "
                 f"{len(prod_mrd_cols)+len(ald_mrd_cols)} targeted mordred] + 4 g-xTB-consistent "
                 f"BDE/BDFE: {', '.join(BDE_COLS)})\n")
        fh.write(f"- **TEST MAE {mae:.3f}, RMSE {rmse:.3f}, R2 {r2:.3f}** vs mordredslim271 production "
                 f"champion (test MAE 1.525) and the quick 2-member XGB check that motivated this run "
                 f"(1.612 -> 1.563, REPORT_bdfe_gxtb_full_augment_20260706.md)\n")
        fh.write(f"- Uncertainty routing: confident {conf.mean()*100:.0f}% MAE {err[conf].mean():.3f} | "
                 f"routed MAE {err[~conf].mean():.3f} | ROC AUC {roc_auc:.3f}\n")
        fh.write(f"- Model: `{bundle}`\n- Full-library output: `products_dG_corrected_{NAME}_{TAG}.csv` "
                 f"({len(fl):,} mols; {int(fl.route_to_dft.sum()):,} flagged route_to_dft)\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
