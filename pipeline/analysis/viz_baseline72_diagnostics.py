#!/usr/bin/env python
"""Full diagnostics for the baseline_72 (champion-equivalent) g-xTB->DFT correction model:
parity/residuals, MLP+XGB training curves, an uncertainty-routing ROC (does the quantile-PI
width predict which molecules will have large error?), error-vs-size, and 2D structure grids
of the worst-mispredicted test-set aldehydes (donor side). One standalone PNG per figure (no
composite panels). Trains fresh (the saved bundle doesn't retain training-history curves).

Outputs -> data/cross_benzoin/homo_v6/viz_gxtb_20260625/ :
  40_parity_baseline72.png, 41_residual_hist.png, 42_residual_vs_pred.png,
  43_mlp_loss_curve.png, 44_xgb_d8_curve.png, 45_xgb_d10_curve.png,
  46_uncertainty_roc.png, 47_error_vs_uncertainty.png, 48_error_vs_molwt.png,
  49_worst_mispredicted_aldehydes.png
  test_predictions_baseline72_<TAG>.csv (full test set)
  worst_mispredictions_baseline72_<TAG>.csv (top 50 by |error|)
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d")
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


def savefig(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print("wrote", name, flush=True)


def main():
    cons = Path(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet")
    dft = pd.read_parquet(cons, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")

    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id", "smiles"] + ALD, low_memory=False).drop_duplicates("id")
    a_r = a.rename(columns={"id": "ald_id", "smiles": "ald_smiles", **{c: f"ald_{c}" for c in ALD}})
    cls = pd.read_parquet(f"{H}/aldehyde_class.parquet")

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a_r, on="ald_id", how="left")
    full = add_global(full, "smiles")
    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print(f"labeled (72-feat) {len(df):,}", flush=True)

    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    sc = StandardScaler().fit(df[FEATS].values[tr])
    Xtr, Xva, Xte = sc.transform(df[FEATS].values[tr]), sc.transform(df[FEATS].values[va]), sc.transform(df[FEATS].values[te])
    dtr, dva = df.delta.values[tr], df.delta.values[va]
    gte, yte = df.dG_gxtb_kcal.values[te], df.dG_orca_kcal.values[te]

    # ── train members, keeping training-history curves ──────────────────────
    mlp = MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4, max_iter=250,
                       early_stopping=True, n_iter_no_change=12, validation_fraction=0.1)
    mlp.fit(Xtr, dtr)

    def xgb(depth, ne):
        return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,
                            colsample_bytree=0.7, min_child_weight=5, n_jobs=16,
                            early_stopping_rounds=60, eval_metric="mae")
    xgb8, xgb10 = xgb(8, 1500), xgb(10, 2000)
    xgb8.fit(Xtr, dtr, eval_set=[(Xtr, dtr), (Xva, dva)], verbose=False)
    xgb10.fit(Xtr, dtr, eval_set=[(Xtr, dtr), (Xva, dva)], verbose=False)

    preds_te = [mlp.predict(Xte), xgb8.predict(Xte), xgb10.predict(Xte)]
    pred = np.vstack(preds_te).mean(0)

    quant = {q: XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, n_estimators=800,
                             max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, n_jobs=16)
             for q in (0.05, 0.95)}
    for q, m in quant.items(): m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
    unc = quant[0.95].predict(Xte) - quant[0.05].predict(Xte)

    yhat = gte + pred; err = np.abs(yhat - yte)
    mae = float(err.mean()); rmse = float(np.sqrt(((yhat - yte) ** 2).mean()))
    r2 = float(1 - ((yhat - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
    thr = float(np.quantile(unc, 1 - ROUTE_FRAC))
    print(f"baseline_72 test MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}", flush=True)

    # ── 1) parity ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, yhat, s=4, alpha=0.25, color="#2171b5")
    lo, hi = min(yte.min(), yhat.min()), max(yte.max(), yhat.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("DFT dG_orca (kcal/mol)"); ax.set_ylabel("corrected prediction (kcal/mol)")
    ax.set_title(f"baseline_72 parity (test, n={len(yte):,})\nMAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    savefig(fig, "40_parity_baseline72.png")

    # ── 2) residual histogram ────────────────────────────────────────────
    resid = yhat - yte
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(resid, bins=80, color="#6baed6", edgecolor="none")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("prediction - DFT (kcal/mol)"); ax.set_ylabel("count")
    ax.set_title(f"baseline_72 residual distribution\nmean={resid.mean():.3f} std={resid.std():.3f}")
    savefig(fig, "41_residual_hist.png")

    # ── 3) residual vs predicted (heteroscedasticity) ───────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(yhat, resid, s=4, alpha=0.25, color="#cb181d")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("predicted dG (kcal/mol)"); ax.set_ylabel("residual (pred - DFT)")
    ax.set_title("baseline_72 residual vs predicted")
    savefig(fig, "42_residual_vs_pred.png")

    # ── 4) MLP training loss curve ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mlp.loss_curve_, label="train loss")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        ax2 = ax.twinx()
        ax2.plot(mlp.validation_scores_, color="#cb181d", label="val R2")
        ax2.set_ylabel("validation R2", color="#cb181d")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title(f"MLP training curve (stopped at epoch {len(mlp.loss_curve_)})")
    savefig(fig, "43_mlp_loss_curve.png")

    # ── 5-6) XGB training curves ─────────────────────────────────────────
    for m, name, fname in [(xgb8, "XGB_d8", "44_xgb_d8_curve.png"), (xgb10, "XGB_d10", "45_xgb_d10_curve.png")]:
        ev = m.evals_result()
        tr_curve = ev["validation_0"]["mae"]; va_curve = ev["validation_1"]["mae"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tr_curve, label="train MAE"); ax.plot(va_curve, label="val MAE")
        ax.axvline(m.best_iteration, color="k", ls="--", lw=1, label=f"best_iter={m.best_iteration}")
        ax.set_xlabel("boosting round"); ax.set_ylabel("MAE (kcal/mol)")
        ax.set_title(f"{name} training curve"); ax.legend()
        savefig(fig, fname)

    # ── 7) uncertainty-routing ROC: does PI width predict large error? ──
    # positive class = test molecule in the worst ROUTE_FRAC by true |error|
    err_thr = float(np.quantile(err, 1 - ROUTE_FRAC))
    y_true_bin = (err >= err_thr).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, unc)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2171b5", lw=2, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"uncertainty-routing ROC\n(does PI width flag the worst {ROUTE_FRAC*100:.0f}% by true error?)")
    ax.legend(loc="lower right")
    savefig(fig, "46_uncertainty_roc.png")

    # ── 8) error vs uncertainty scatter ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    conf = unc < thr
    ax.scatter(unc[conf], err[conf], s=4, alpha=0.3, color="#2171b5", label=f"confident ({conf.mean()*100:.0f}%)")
    ax.scatter(unc[~conf], err[~conf], s=4, alpha=0.3, color="#cb181d", label=f"routed ({(~conf).mean()*100:.0f}%)")
    ax.axvline(thr, color="k", ls="--", lw=1)
    ax.set_xlabel("quantile PI width (uncertainty)"); ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title(f"error vs uncertainty  (conf MAE={err[conf].mean():.3f}, routed MAE={err[~conf].mean():.3f})")
    ax.legend()
    savefig(fig, "47_error_vs_uncertainty.png")

    # ── 9) error vs molecular size ───────────────────────────────────────
    molwt = df["g_MolWt"].values[te]
    scope_te = df["cls"].values[te]
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, c in [("aromatic", "#2171b5"), ("aliphatic", "#cb181d")]:
        mk = scope_te == s
        ax.scatter(molwt[mk], err[mk], s=4, alpha=0.3, color=c, label=s)
    ax.set_xlabel("product MolWt (g/mol)"); ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title("baseline_72 error vs molecular size")
    ax.legend()
    savefig(fig, "48_error_vs_molwt.png")

    # ── test-set predictions CSV ─────────────────────────────────────────
    te_df = df.iloc[te].reset_index(drop=True).copy()
    te_df["dG_pred"] = yhat; te_df["error"] = err; te_df["uncertainty_pi_width"] = unc
    te_df["route_to_dft"] = ~conf
    keep = ["id", "ald_id", "smiles", "ald_smiles", "cls", "dG_orca_kcal", "dG_gxtb_kcal",
            "dG_pred", "error", "uncertainty_pi_width", "route_to_dft"]
    te_df[keep].to_csv(OUT / f"test_predictions_baseline72_{TAG}.csv", index=False)
    print(f"wrote test_predictions_baseline72_{TAG}.csv ({len(te_df):,} rows)", flush=True)

    # ── worst mispredicted aldehydes: structure grid + CSV ─────────────
    worst = te_df.sort_values("error", ascending=False).head(50).reset_index(drop=True)
    worst[keep].to_csv(OUT / f"worst_mispredictions_baseline72_{TAG}.csv", index=False)
    print(f"wrote worst_mispredictions_baseline72_{TAG}.csv (top 50)", flush=True)

    top20 = worst.head(20)
    mols, legends = [], []
    for _, r in top20.iterrows():
        m = Chem.MolFromSmiles(str(r["ald_smiles"]))
        if m is None:
            continue
        mols.append(m)
        legends.append(f"id={r['id']} err={r['error']:.2f}\npred={r['dG_pred']:.1f} true={r['dG_orca_kcal']:.1f}")
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(260, 220), legends=legends)
        img.save(OUT / "49_worst_mispredicted_aldehydes.png")
        print("wrote 49_worst_mispredicted_aldehydes.png", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
