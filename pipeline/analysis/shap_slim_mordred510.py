#!/usr/bin/env python
"""SHAP-based importance + correlation slimming of the mordred510 champion (per
descriptor-slim-v4 methodology: importance + correlation pruning, NOT PCA -- user's
standing preference, preserves interpretability of every kept feature).

Loads the SAVED bundle (no retraining) to reproduce the exact test split, computes SHAP
(XGB_d8, TreeExplainer) for global importance, then greedily prunes the 438 mordred feats:
keep in descending |SHAP| order, drop any feature with |corr|>0.9 to an already-kept one
(the 72 champion QM feats are never touched -- they're the validated baseline).
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import joblib, shap
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d")

TARGET_MODULES = {"MoRSE", "CPSA", "Polarizability", "GeometricalIndex", "MomentOfInertia",
                  "PBF", "McGowanVolume", "VdwVolumeABC", "Weight", "TopoPSA"}


def targeted_mordred_names() -> set[str]:
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=False)
    return {str(d) for d in calc.descriptors if type(d).__module__.split(".")[-1] in TARGET_MODULES}


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


def savefig(name):
    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", name, flush=True)


def main():
    bundle = joblib.load(f"{R}/pipeline/models/gxtb_dft_correction_MORDRED510_20260703.joblib")
    FEATS = bundle["features"]; imp = bundle["imputer"]; sc = bundle["scaler"]
    xgb8 = dict(bundle["members"])["XGB_d8"]
    mordred_feats = [f for f in FEATS if f not in FEATS_72]
    print(f"bundle: {len(FEATS)} feats ({len(FEATS_72)} QM + {len(mordred_feats)} mordred)", flush=True)

    # ── reproduce the exact data + split used to train this bundle ──────────
    cons = Path(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet")
    dft = pd.read_parquet(cons, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id"] + ALD, low_memory=False).drop_duplicates("id").rename(columns={"id": "ald_id", **{c: f"ald_{c}" for c in ALD}})
    cls = pd.read_parquet(f"{H}/aldehyde_class.parquet")

    target_names = targeted_mordred_names()
    prod_header = pd.read_csv(f"{H}/products_mordred_descriptors.csv", nrows=0).columns
    prod_want = ["id"] + [c for c in prod_header if c.replace("mordred_", "") in target_names]
    prod_mrd = pd.read_csv(f"{H}/products_mordred_descriptors.csv", usecols=prod_want, low_memory=False)
    ald_header = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", nrows=0).columns
    ald_want = ["id"] + [c for c in ald_header if c.replace("mordred_", "") in target_names]
    ald_mrd = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", usecols=ald_want, low_memory=False)
    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith("mordred_") and c in mordred_feats]
    ald_mrd = ald_mrd.rename(columns={"id": "ald_id"})
    ald_mrd_raw = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={c: f"ald_{c}" for c in ald_mrd_raw})
    ald_mrd_cols = [f"ald_{c}" for c in ald_mrd_raw if f"ald_{c}" in mordred_feats]
    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")
    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS_72).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print(f"labeled rows: {len(df):,} (bundle trained on {bundle['n_train']:,})", flush=True)

    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    Xtr = sc.transform(imp.transform(df[FEATS].values[tr]))
    Xte = sc.transform(imp.transform(df[FEATS].values[te]))
    Xte_df = pd.DataFrame(Xte, columns=FEATS)

    # ── SHAP on a subsample (union pattern from shap_baseline72.py) ─────────
    pred8 = xgb8.predict(Xte)
    err8 = np.abs(pred8 - (df.dG_orca_kcal.values[te] - df.dG_gxtb_kcal.values[te]))
    rng2 = np.random.default_rng(7)
    sub_n = min(4000, len(Xte))
    sub_idx = rng2.choice(len(Xte), size=sub_n, replace=False)
    d8_frame = Xte_df.iloc[sub_idx].reset_index(drop=True)
    expl8 = shap.TreeExplainer(xgb8)
    sv8 = expl8(d8_frame)
    print(f"SHAP computed on {sub_n} test-subsample rows", flush=True)

    mean_abs = pd.Series(np.abs(sv8.values).mean(0), index=FEATS).sort_values(ascending=False)
    mean_abs.to_csv(OUT / f"shap_importance_mordred510_{TAG}.csv", header=["mean_abs_shap"])

    # global summary for the mordred subset only (72 QM already shown in shap_baseline72)
    mrd_idx = [FEATS.index(f) for f in mordred_feats]
    shap.summary_plot(sv8.values[:, mrd_idx], d8_frame[mordred_feats], show=False, max_display=30)
    savefig(f"80_shap_summary_mordred_only_{TAG}.png")

    # ── greedy importance + correlation pruning of the 438 mordred feats ────
    corr = df[mordred_feats].iloc[tr].corr().abs()
    ranked_mordred = [f for f in mean_abs.index if f in mordred_feats]
    kept, dropped = [], []
    for f in ranked_mordred:
        if any(corr.loc[f, k] > 0.9 for k in kept):
            dropped.append(f)
        else:
            kept.append(f)
    print(f"mordred slim: kept {len(kept)}/{len(mordred_feats)} (dropped {len(dropped)} as "
         f"redundant with a higher-importance feature, |corr|>0.9)", flush=True)

    slim_feats = FEATS_72 + kept
    json.dump({"kept_mordred": kept, "dropped_mordred": dropped, "feats_72": FEATS_72,
              "slim_feats_total": slim_feats}, open(OUT / f"mordred_slim_selection_{TAG}.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(9, 8))
    top25 = mean_abs.head(25)
    colors = ["#cb181d" if f in FEATS_72 else "#2171b5" for f in top25.index]
    ax.barh(range(len(top25)), top25.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top25))); ax.set_yticklabels(top25.index[::-1], fontsize=8)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("mordred510 top-25 global importance\n(red=original 72 QM, blue=mordred)")
    savefig(f"81_top25_importance_mordred510_{TAG}.png")

    rep = OUT / f"REPORT_mordred_slim_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# mordred510 SHAP importance + correlation slim ({TAG})\n\n")
        fh.write(f"Bundle: `gxtb_dft_correction_MORDRED510_20260703.joblib` (test MAE "
                 f"{bundle['test_mae']:.3f}). SHAP on {sub_n}-row test subsample (XGB_d8).\n\n")
        fh.write(f"## Slimming result\n\nkept **{len(kept)}/{len(mordred_feats)}** mordred feats "
                 f"(dropped {len(dropped)} redundant, |corr|>0.9 with a higher-ranked kept feat). "
                 f"72 champion QM feats never touched.\n\n**Slim total: {len(slim_feats)} feats** "
                 f"(72 + {len(kept)}).\n\n")
        fh.write("## Top-25 global importance\n\n| rank | feature | mean|SHAP| | family |\n|---|---|---|---|\n")
        for i, (f, v) in enumerate(mean_abs.head(25).items(), 1):
            fam = "QM(72)" if f in FEATS_72 else "mordred"
            fh.write(f"| {i} | {f} | {v:.4f} | {fam} |\n")
        fh.write(f"\nSelection saved: `mordred_slim_selection_{TAG}.json` (use `slim_feats_total` "
                 f"to retrain a slimmed model and confirm it holds the MAE 1.517 level with fewer feats).\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
