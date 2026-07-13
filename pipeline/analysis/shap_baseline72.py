#!/usr/bin/env python
"""Deep interpretability (SHAP) analysis for baseline_72, the g-xTB->DFT correction model.

Scope (agreed plan, 2026-07-01):
  1. Global SHAP summary (beeswarm) for the two tree members (XGB_d8, XGB_d10) -- MLP
     skipped (not a tree model; TreeExplainer is exact+fast, KernelExplainer for MLP would
     be orders of magnitude slower for a marginal 3rd view).
  2. SHAP dependence plots for the top-6 gain-importance features (direction + interaction).
  3. Local waterfall explanations for the worst-20 mispredicted test molecules (why did the
     model get THESE wrong, feature by feature).
  4. SHAP-fingerprint clustering of the worst 15% (routed) test molecules -- do the biggest
     errors share a common attribution pattern, or are they heterogeneous failure modes?

Same 70:20:10 split (seed 42) as viz_baseline72_diagnostics.py / finalize_correction.py, so
molecule-level results line up with test_predictions_baseline72_20260701.csv.
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
import shap
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

    def xgb(depth, ne):
        return XGBRegressor(n_estimators=ne, max_depth=depth, learning_rate=0.02, subsample=0.7,
                            colsample_bytree=0.7, min_child_weight=5, n_jobs=16,
                            early_stopping_rounds=60, eval_metric="mae")
    models = {"XGB_d8": xgb(8, 1500), "XGB_d10": xgb(10, 2000)}
    for nm, m in models.items():
        m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
        pred = gte + m.predict(Xte)
        print(f"{nm} standalone test MAE={np.abs(pred - yte).mean():.3f}", flush=True)

    Xte_df = pd.DataFrame(Xte, columns=FEATS)  # standardized values, but SHAP works in this space; feature_names carried
    ids_te = df["id"].values[te]

    # ── cheap predictions first (fast, no SHAP) to size the expensive SHAP calls ──
    pred8 = gte + models["XGB_d8"].predict(Xte)
    err8 = np.abs(pred8 - yte)
    worst_idx = np.argsort(err8)[::-1][:20]
    thr_err = np.quantile(err8, 1 - ROUTE_FRAC)
    routed_idx = np.where(err8 >= thr_err)[0]

    # ── exact Tree SHAP on all 21,910 test rows for a depth-10/2000-tree model timed
    # out at 2h (job 24359381) with only XGB_d8's summary done. Fix: subsample for the
    # expensive global views (summary/dependence), keep exact SHAP only where a specific
    # molecule identity matters (worst-20 waterfalls, routed-15% clustering).
    rng2 = np.random.default_rng(7)
    sub_n = min(3000, len(Xte))
    sub_idx = rng2.choice(len(Xte), size=sub_n, replace=False)
    # union of subsample + routed-15% (waterfall's worst-20 is a subset of routed) -> one
    # XGB_d8 SHAP call covers summary/dependence/clustering/waterfall without recomputation.
    d8_idx = np.union1d(sub_idx, routed_idx)
    print(f"SHAP subsample: {sub_n} for summary/dependence, {len(routed_idx)} routed "
         f"(union {len(d8_idx)}) for XGB_d8; {sub_n} for XGB_d10 summary only", flush=True)

    # ── 1) global SHAP summary (beeswarm) per tree member ────────────────
    # NB: shap.Explanation indexing on a DataFrame-backed explainer call follows the
    # DataFrame's row LABELS, not raw position -- .iloc[...] preserves original (non-
    # contiguous) labels, which breaks `sv[k]`/`sv[list]` lookups downstream. Always
    # reset_index(drop=True) right before handing a sliced frame to TreeExplainer so its
    # Explanation's positions are a clean 0..N-1 range matching our own bookkeeping.
    d8_frame = Xte_df.iloc[d8_idx].reset_index(drop=True)
    expl8 = shap.TreeExplainer(models["XGB_d8"])
    sv8_full = expl8(d8_frame)  # position k <-> d8_idx[k]
    pos_in_d8 = {orig: k for k, orig in enumerate(d8_idx)}  # te-position -> row in sv8_full
    sub_pos = [pos_in_d8[i] for i in sub_idx]
    shap.summary_plot(sv8_full[sub_pos], d8_frame.iloc[sub_pos], show=False, max_display=25)
    savefig("50_shap_summary_XGB_d8.png")

    sub_frame = Xte_df.iloc[sub_idx].reset_index(drop=True)
    expl10 = shap.TreeExplainer(models["XGB_d10"])
    sv10_sub = expl10(sub_frame)
    shap.summary_plot(sv10_sub, sub_frame, show=False, max_display=25)
    savefig("50_shap_summary_XGB_d10.png")

    # save raw SHAP values (XGB_d8, the deeper/likely-stronger single model per exp5) for reuse
    sv8_df = pd.DataFrame(sv8_full.values, columns=FEATS)
    sv8_df.insert(0, "id", ids_te[d8_idx])
    sv8_df.insert(1, "in_summary_subsample", np.isin(d8_idx, sub_idx))
    sv8_df.insert(2, "in_routed_worst15pct", np.isin(d8_idx, routed_idx))
    sv8_df.to_csv(OUT / f"shap_values_xgb_d8_{TAG}.csv", index=False)
    print(f"wrote shap_values_xgb_d8_{TAG}.csv ({len(sv8_df):,} rows, union of "
         f"{sub_n}-subsample + {len(routed_idx)} routed)", flush=True)

    # ── 2) dependence plots for top-6 |SHAP| features (XGB_d8, subsample) ─
    mean_abs = np.abs(sv8_full.values[sub_pos]).mean(0)
    top6 = [FEATS[i] for i in np.argsort(mean_abs)[::-1][:6]]
    for feat in top6:
        shap.dependence_plot(feat, sv8_full.values[sub_pos], sub_frame, show=False)
        savefig(f"51_shap_dependence_{feat}.png")

    # ── 3) local waterfall for worst-20 mispredicted molecules (exact) ───
    for rank, i in enumerate(worst_idx, 1):
        sv1 = sv8_full[pos_in_d8[i]]
        shap.plots.waterfall(sv1, max_display=15, show=False)
        savefig(f"52_shap_waterfall_rank{rank:02d}_id{ids_te[i]}.png")

    # ── 4) SHAP-fingerprint clustering of the routed (worst 15%) set ────
    routed_pos = [pos_in_d8[i] for i in routed_idx]
    sv_routed = sv8_full.values[routed_pos]
    n_clusters = min(4, len(routed_idx) // 20 or 1)
    if n_clusters >= 2:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(sv_routed)
        fig, ax = plt.subplots(figsize=(9, 6))
        for c in range(n_clusters):
            mk = km.labels_ == c
            prof = sv_routed[mk].mean(0)
            top = np.argsort(np.abs(prof))[::-1][:8]
            ax.barh([f"{FEATS[j]} (c{c})" for j in top], prof[top],
                   alpha=0.6, label=f"cluster {c} (n={mk.sum()})")
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("mean SHAP value in cluster")
        ax.set_title(f"SHAP-fingerprint clusters of worst {ROUTE_FRAC*100:.0f}% test molecules")
        ax.legend(fontsize=7)
        savefig("53_shap_fingerprint_clusters.png")
        clus_df = pd.DataFrame({"id": ids_te[routed_idx], "error": err8[routed_idx], "cluster": km.labels_})
        clus_df.to_csv(OUT / f"shap_fingerprint_clusters_{TAG}.csv", index=False)
        print(f"wrote shap_fingerprint_clusters_{TAG}.csv", flush=True)
    else:
        print("too few routed molecules for clustering, skipped step 4", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
