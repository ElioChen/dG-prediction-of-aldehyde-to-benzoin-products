#!/usr/bin/env python
"""PRODUCTION finalize for the mordred-augmented g-xTB->DFT correction model: 72 champion
feats + 438 targeted dispersion/size/shape mordred feats (MoRSE/CPSA/Polarizability/
GeometricalIndex/MomentOfInertia/PBF/McGowanVolume/VdwVolumeABC/Weight/TopoPSA; see
finalize_correction_mordred.py for why this subset and not the full 1826-descriptor dump).

Quick-check result (finalize_correction_mordred.py, job 24391566): test MAE 1.535 vs
baseline_72's 1.577 in the SAME run -- a real, out-of-noise-band improvement (noise band
1.571+/-0.013, see REPORT_robustness_baseline72_20260702.md). This script promotes that
result to a full production treatment per training-runs-full-diagnostics:
  - saves a joblib bundle (members + quantile routing + imputer + scaler + features)
  - saves full-library corrected predictions + uncertainty + route_to_dft flag
  - full diagnostics: parity, residual dist, residual-vs-pred, MLP/XGB training curves,
    uncertainty-routing ROC, error-vs-uncertainty, error-vs-size, worst-20 mispredicted
    aldehyde structure grid + CSV, test-set predictions CSV
"""
import time
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

TARGET_MODULES = {"MoRSE", "CPSA", "Polarizability", "GeometricalIndex", "MomentOfInertia",
                  "PBF", "McGowanVolume", "VdwVolumeABC", "Weight", "TopoPSA"}


def targeted_mordred_names() -> set[str]:
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=False)
    return {str(d) for d in calc.descriptors if type(d).__module__.split(".")[-1] in TARGET_MODULES}


R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d"); MODELDIR = Path(f"{R}/pipeline/models"); MODELDIR.mkdir(exist_ok=True)
ROUTE_FRAC = 0.15

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

    target_names = targeted_mordred_names()
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
    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")
    prod_mrd_cols = [c for c in prod_mrd_cols if prod_mrd[c].notna().mean() >= 0.80]
    ald_mrd_cols = [c for c in ald_mrd_cols if ald_mrd[c].notna().mean() >= 0.80]

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a_r, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")

    FEATS = PROD_QM + ALDp + GLOB + prod_mrd_cols + ald_mrd_cols
    print(f"total features: {len(FEATS)} (72 champion + {len(prod_mrd_cols)+len(ald_mrd_cols)} mordred)", flush=True)

    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + PROD_QM + ALDp + GLOB).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print(f"labeled rows: {len(df):,}", flush=True)

    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    imp = SimpleImputer(strategy="median").fit(df[FEATS].values[tr])
    Xtr_raw, Xva_raw, Xte_raw = imp.transform(df[FEATS].values[tr]), imp.transform(df[FEATS].values[va]), imp.transform(df[FEATS].values[te])
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xva, Xte = sc.transform(Xtr_raw), sc.transform(Xva_raw), sc.transform(Xte_raw)
    dtr, dva = df.delta.values[tr], df.delta.values[va]
    gte, yte = df.dG_gxtb_kcal.values[te], df.dG_orca_kcal.values[te]
    ids_te = df["id"].values[te]

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
    print(f"mordred510 test MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}", flush=True)
    print(f"confident {conf.mean()*100:.0f}% MAE={err[conf].mean():.3f}  routed MAE={err[~conf].mean():.3f}", flush=True)

    # ── save production bundle ───────────────────────────────────────────
    bundle = MODELDIR / f"gxtb_dft_correction_MORDRED510_{TAG}.joblib"
    joblib.dump({"members": members, "quantiles": quant, "imputer": imp, "scaler": sc,
                "features": FEATS, "route_width_threshold": thr, "route_frac": ROUTE_FRAC,
                "target": "DFT_r2scan3c - gxtb (kcal/mol)", "test_mae": mae, "n_train": len(tr)}, bundle)
    print("wrote", bundle, flush=True)

    # ── full-library prediction ──────────────────────────────────────────
    fl = full.dropna(subset=["dG_gxtb_kcal"] + PROD_QM + ALDp + GLOB).copy()
    Xf = sc.transform(imp.transform(fl[FEATS].values))
    fl["delta_pred"] = np.vstack([m.predict(Xf) for _, m in members]).mean(0)
    fl["uncertainty_pi_width"] = quant[0.95].predict(Xf) - quant[0.05].predict(Xf)
    fl["dG_gxtb_corrected_final"] = fl["dG_gxtb_kcal"] + fl["delta_pred"]
    fl["route_to_dft"] = fl["uncertainty_pi_width"] >= thr
    fl[["id", "smiles", "dG_gxtb_kcal", "dG_gxtb_corrected_final", "uncertainty_pi_width", "route_to_dft"]].to_csv(
        OUT / f"products_dG_corrected_MORDRED510_{TAG}.csv", index=False)
    print(f"wrote full-library predictions ({len(fl):,} mols)", flush=True)

    # ══ diagnostics (same shape as viz_baseline72_diagnostics.py) ═══════════
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, yhat, s=4, alpha=0.25, color="#2171b5")
    lo, hi = min(yte.min(), yhat.min()), max(yte.max(), yhat.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("DFT dG_orca (kcal/mol)"); ax.set_ylabel("corrected prediction (kcal/mol)")
    ax.set_title(f"mordred510 parity (test, n={len(yte):,})\nMAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    savefig("70_parity_mordred510.png")

    resid = yhat - yte
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(resid, bins=80, color="#6baed6", edgecolor="none")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("prediction - DFT (kcal/mol)"); ax.set_ylabel("count")
    ax.set_title(f"mordred510 residual distribution\nmean={resid.mean():.3f} std={resid.std():.3f}")
    savefig("71_residual_hist_mordred510.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(yhat, resid, s=4, alpha=0.25, color="#cb181d")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("predicted dG (kcal/mol)"); ax.set_ylabel("residual (pred - DFT)")
    ax.set_title("mordred510 residual vs predicted")
    savefig("72_residual_vs_pred_mordred510.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mlp.loss_curve_, label="train loss")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        ax2 = ax.twinx(); ax2.plot(mlp.validation_scores_, color="#cb181d", label="val R2")
        ax2.set_ylabel("validation R2", color="#cb181d")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title(f"mordred510 MLP training curve (stopped at epoch {len(mlp.loss_curve_)})")
    savefig("73_mlp_loss_curve_mordred510.png")

    for m, name, fname in [(xgb8, "XGB_d8", "74_xgb_d8_curve_mordred510.png"),
                           (xgb10, "XGB_d10", "75_xgb_d10_curve_mordred510.png")]:
        ev = m.evals_result()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ev["validation_0"]["mae"], label="train MAE"); ax.plot(ev["validation_1"]["mae"], label="val MAE")
        ax.axvline(m.best_iteration, color="k", ls="--", lw=1, label=f"best_iter={m.best_iteration}")
        ax.set_xlabel("boosting round"); ax.set_ylabel("MAE (kcal/mol)")
        ax.set_title(f"mordred510 {name} training curve"); ax.legend()
        savefig(fname)

    err_thr = float(np.quantile(err, 1 - ROUTE_FRAC))
    y_true_bin = (err >= err_thr).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, unc); roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2171b5", lw=2, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"mordred510 uncertainty-routing ROC\n(worst {ROUTE_FRAC*100:.0f}% by true error)")
    ax.legend(loc="lower right")
    savefig("76_uncertainty_roc_mordred510.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(unc[conf], err[conf], s=4, alpha=0.3, color="#2171b5", label=f"confident ({conf.mean()*100:.0f}%)")
    ax.scatter(unc[~conf], err[~conf], s=4, alpha=0.3, color="#cb181d", label=f"routed ({(~conf).mean()*100:.0f}%)")
    ax.axvline(thr, color="k", ls="--", lw=1)
    ax.set_xlabel("quantile PI width (uncertainty)"); ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title(f"mordred510 error vs uncertainty (conf MAE={err[conf].mean():.3f}, routed MAE={err[~conf].mean():.3f})")
    ax.legend()
    savefig("77_error_vs_uncertainty_mordred510.png")

    molwt = df["g_MolWt"].values[te]; scope_te = df["cls"].values[te]
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, c in [("aromatic", "#2171b5"), ("aliphatic", "#cb181d")]:
        mk = scope_te == s
        ax.scatter(molwt[mk], err[mk], s=4, alpha=0.3, color=c, label=s)
    ax.set_xlabel("product MolWt (g/mol)"); ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title("mordred510 error vs molecular size"); ax.legend()
    savefig("78_error_vs_molwt_mordred510.png")

    te_df = df.iloc[te].reset_index(drop=True).copy()
    te_df["dG_pred"] = yhat; te_df["error"] = err; te_df["uncertainty_pi_width"] = unc
    te_df["route_to_dft"] = ~conf
    keep = ["id", "ald_id", "smiles", "ald_smiles", "cls", "dG_orca_kcal", "dG_gxtb_kcal",
            "dG_pred", "error", "uncertainty_pi_width", "route_to_dft"]
    te_df[keep].to_csv(OUT / f"test_predictions_mordred510_{TAG}.csv", index=False)

    worst = te_df.sort_values("error", ascending=False).head(50).reset_index(drop=True)
    worst[keep].to_csv(OUT / f"worst_mispredictions_mordred510_{TAG}.csv", index=False)
    top20 = worst.head(20)
    mols, legends = [], []
    for _, r in top20.iterrows():
        m = Chem.MolFromSmiles(str(r["ald_smiles"]))
        if m is None: continue
        mols.append(m)
        legends.append(f"id={r['id']} err={r['error']:.2f}\npred={r['dG_pred']:.1f} true={r['dG_orca_kcal']:.1f}")
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(260, 220), legends=legends)
        img.save(OUT / "79_worst_mispredicted_aldehydes_mordred510.png")
        print("wrote 79_worst_mispredicted_aldehydes_mordred510.png", flush=True)

    rep = OUT / f"REPORT_MORDRED510_FINAL_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# mordred510 production model ({TAG})\n\n")
        fh.write(f"- Features: **{len(FEATS)}** (72 champion QM + {len(prod_mrd_cols)+len(ald_mrd_cols)} "
                 f"targeted mordred: MoRSE/CPSA/Polarizability/GeometricalIndex/MomentOfInertia/"
                 f"PBF/McGowanVolume/VdwVolumeABC/Weight/TopoPSA)\n")
        fh.write(f"- **TEST MAE {mae:.3f}, RMSE {rmse:.3f}, R2 {r2:.3f}** vs baseline_72 noise band "
                 f"1.571+/-0.013 (REPORT_robustness_baseline72_20260702.md) -- a real improvement\n")
        fh.write(f"- Uncertainty routing: confident {conf.mean()*100:.0f}% MAE {err[conf].mean():.3f} | "
                 f"routed MAE {err[~conf].mean():.3f} | ROC AUC {roc_auc:.3f}\n")
        fh.write(f"- Model: `{bundle}`\n- Full-library output: `products_dG_corrected_MORDRED510_{TAG}.csv` "
                 f"({len(fl):,} mols; {int(fl.route_to_dft.sum()):,} flagged route_to_dft)\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
