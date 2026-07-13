#!/usr/bin/env python
"""SHAP-based importance + COST-AWARE slimming of the current champion
(MORDREDSLIM271_BDEGXTB, 275 feats, test MAE 1.503, job 24468737).

Two questions this answers:
1. Standard descriptor_slim_v4 methodology (importance + correlation pruning, NOT PCA):
   can the 199 mordred feats be pruned further without hurting accuracy?
2. NEW: is the expensive BDFE (full xtb --ohess Hessian + Shermo qRRHO thermal correction,
   ~2x the cost of BDE per molecule since it needs 2 fragment ohess calcs, vs BDE's single
   cheap SP/opt) actually pulling its weight in SHAP importance, or is nearly all of the
   4-feature BDE/BDFE gain (mordredslim271 1.525 -> 1.503) coming from the 2 cheap BDE
   columns alone? If BDFE's importance is negligible, a future cheaper "BDE-only" variant
   (277->273 feat drop) could be tested to see if it holds ~1.50 without ever needing the
   expensive Hessian calc for new molecules -- material for prospective screening cost.

Loads the SAVED bundle (no retraining) to reproduce the exact test split, computes SHAP
(XGB_d8, TreeExplainer) for global importance.
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
# rough per-molecule compute-cost tier, for the cost-vs-importance framing (not used in the
# math, just annotates the report): BDE = 1 extra xtb SP/opt per side (cheap); BDFE = full
# --ohess (Hessian + RRHO) per side, ~much more expensive, see bde-descriptor-idea memory
COST_TIER = {"prod_bde_gxtb_kcal": "cheap (SP/opt)", "ald_bde_gxtb_kcal": "cheap (SP/opt)",
             "prod_bdfe_gxtb_kcal": "EXPENSIVE (--ohess Hessian+RRHO)",
             "ald_bdfe_gxtb_kcal": "EXPENSIVE (--ohess Hessian+RRHO)"}


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
    bundle = joblib.load(f"{R}/pipeline/models/gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib")
    FEATS = bundle["features"]; imp = bundle["imputer"]; sc = bundle["scaler"]
    xgb8 = dict(bundle["members"])["XGB_d8"]
    mordred_feats = [f for f in FEATS if f not in FEATS_72 and f not in BDE_COLS]
    print(f"bundle: {len(FEATS)} feats ({len(FEATS_72)} QM + {len(mordred_feats)} mordred + {len(BDE_COLS)} BDE/BDFE), test_mae={bundle['test_mae']:.3f}", flush=True)

    # ── reproduce the exact data + split used to train this bundle ──────────
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
    prod_mrd_cols = [c for c in prod_mrd.columns if c.startswith("mordred_") and c in mordred_feats]
    ald_mrd = ald_mrd.rename(columns={"id": "ald_id"})
    ald_mrd_raw = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    ald_mrd = ald_mrd.rename(columns={c: f"ald_{c}" for c in ald_mrd_raw})
    ald_mrd_cols = [f"ald_{c}" for c in ald_mrd_raw if f"ald_{c}" in mordred_feats]
    for c in prod_mrd_cols: prod_mrd[c] = pd.to_numeric(prod_mrd[c], errors="coerce")
    for c in ald_mrd_cols: ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")

    prod_bde = pd.read_csv(f"{H}/products_bdfe_gxtb_descriptors.csv",
                            usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"bdfe_gxtb_kcal": "prod_bdfe_gxtb_kcal", "bde_gxtb_kcal": "prod_bde_gxtb_kcal"})
    ald_bde = pd.read_csv(f"{H}/aldehydes_bdfe_gxtb_descriptors.csv",
                           usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"id": "ald_id", "bdfe_gxtb_kcal": "ald_bdfe_gxtb_kcal", "bde_gxtb_kcal": "ald_bde_gxtb_kcal"})
    for c in ["prod_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal"]: prod_bde.loc[prod_bde[c].abs() > 200, c] = np.nan
    for c in ["ald_bdfe_gxtb_kcal", "ald_bde_gxtb_kcal"]: ald_bde.loc[ald_bde[c].abs() > 200, c] = np.nan

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")
    full = full.merge(prod_bde, on="id", how="left")
    full = full.merge(ald_bde, on="ald_id", how="left")

    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS_72).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print(f"labeled rows: {len(df):,} (bundle trained on {bundle['n_train']:,})", flush=True)

    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); tr, va, te = idx[:ntr], idx[ntr:nva], idx[nva:]
    Xte = sc.transform(imp.transform(df[FEATS].values[te]))
    Xte_df = pd.DataFrame(Xte, columns=FEATS)

    # ── SHAP on a test subsample ─────────────────────────────────────────
    rng2 = np.random.default_rng(7)
    sub_n = min(4000, len(Xte))
    sub_idx = rng2.choice(len(Xte), size=sub_n, replace=False)
    d8_frame = Xte_df.iloc[sub_idx].reset_index(drop=True)
    expl8 = shap.TreeExplainer(xgb8)
    sv8 = expl8(d8_frame)
    print(f"SHAP computed on {sub_n} test-subsample rows", flush=True)

    mean_abs = pd.Series(np.abs(sv8.values).mean(0), index=FEATS).sort_values(ascending=False)
    mean_abs.to_csv(OUT / f"shap_importance_mordredslim271_bdegxtb_{TAG}.csv", header=["mean_abs_shap"])

    # ── BDE/BDFE-specific cost-vs-importance readout ────────────────────────
    bde_rank = {f: (int(mean_abs.index.get_loc(f)) + 1, float(mean_abs[f])) for f in BDE_COLS}
    n_total = len(FEATS)
    print("BDE/BDFE importance ranks (1=most important, out of "
          f"{n_total}):", flush=True)
    for f, (rk, v) in bde_rank.items():
        print(f"  {f:26s} rank {rk:4d}/{n_total}  mean|SHAP|={v:.4f}  cost={COST_TIER[f]}", flush=True)
    bde_only_sum = bde_rank["prod_bde_gxtb_kcal"][1] + bde_rank["ald_bde_gxtb_kcal"][1]
    bdfe_only_sum = bde_rank["prod_bdfe_gxtb_kcal"][1] + bde_rank["ald_bdfe_gxtb_kcal"][1]
    print(f"summed mean|SHAP|: BDE(cheap)={bde_only_sum:.4f}  BDFE(expensive)={bdfe_only_sum:.4f}  "
          f"ratio BDFE/BDE={bdfe_only_sum/max(bde_only_sum,1e-9):.2f}", flush=True)

    # dependence plots for the 4 new features specifically
    for f in BDE_COLS:
        fi = FEATS.index(f)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(d8_frame[f], sv8.values[:, fi], s=6, alpha=0.35, color="#2171b5")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel(f); ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP dependence: {f}\n({COST_TIER[f]})")
        savefig(f"110_shap_dependence_{f}.png")

    # global summary limited to the 4 new BDE/BDFE feats + top mordred, for visual context
    top_idx = [FEATS.index(f) for f in list(mean_abs.head(20).index)]
    shap.summary_plot(sv8.values[:, top_idx], d8_frame[[FEATS[i] for i in top_idx]], show=False, max_display=20)
    savefig(f"111_shap_summary_top20_{TAG}.png")

    # ── greedy importance + correlation pruning of mordred+BDE/BDFE feats (BDE/BDFE never
    # dropped for redundancy -- they're mechanistically distinct, only mordred gets pruned) ──
    prunable = mordred_feats
    corr = df[prunable].iloc[tr].corr().abs()
    ranked = [f for f in mean_abs.index if f in prunable]
    kept, dropped = [], []
    for f in ranked:
        if any(corr.loc[f, k] > 0.9 for k in kept):
            dropped.append(f)
        else:
            kept.append(f)
    print(f"mordred re-slim (on top of existing 199): kept {len(kept)}/{len(prunable)} "
          f"(dropped {len(dropped)} redundant, |corr|>0.9)", flush=True)

    slim_feats = FEATS_72 + kept + BDE_COLS
    cost_aware_feats = FEATS_72 + kept + ["prod_bde_gxtb_kcal", "ald_bde_gxtb_kcal"]  # drop expensive BDFE
    json.dump({"kept_mordred": kept, "dropped_mordred": dropped, "feats_72": FEATS_72,
              "bde_cols": BDE_COLS, "slim_feats_total": slim_feats,
              "cost_aware_feats_total": cost_aware_feats,
              "bde_bdfe_shap_rank": {k: v[0] for k, v in bde_rank.items()},
              "bde_bdfe_mean_abs_shap": {k: v[1] for k, v in bde_rank.items()}},
              open(OUT / f"mordredslim271_bdegxtb_slim_selection_{TAG}.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(9, 8))
    top25 = mean_abs.head(25)
    def color_for(f):
        if f in BDE_COLS: return "#238b45"
        if f in FEATS_72: return "#cb181d"
        return "#2171b5"
    colors = [color_for(f) for f in top25.index]
    ax.barh(range(len(top25)), top25.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top25))); ax.set_yticklabels(top25.index[::-1], fontsize=8)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("MORDREDSLIM271_BDEGXTB top-25 global importance\n(red=72 QM, blue=mordred, green=g-xTB BDE/BDFE)")
    savefig(f"112_top25_importance_mordredslim271_bdegxtb_{TAG}.png")

    rep = OUT / f"REPORT_shap_mordredslim271_bdegxtb_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# MORDREDSLIM271_BDEGXTB SHAP importance + cost-aware slimming ({TAG})\n\n")
        fh.write(f"Bundle: `gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib` "
                 f"(test MAE {bundle['test_mae']:.3f}). SHAP on {sub_n}-row test subsample (XGB_d8).\n\n")
        fh.write("## BDE vs BDFE: is the expensive descriptor pulling its weight?\n\n")
        fh.write("| feature | SHAP rank (of 275) | mean|SHAP| | acquisition cost |\n|---|---|---|---|\n")
        for f in BDE_COLS:
            rk, v = bde_rank[f]
            fh.write(f"| {f} | {rk} | {v:.4f} | {COST_TIER[f]} |\n")
        fh.write(f"\nSummed importance: BDE (cheap, SP/opt only) = {bde_only_sum:.4f}; "
                 f"BDFE (expensive, full `--ohess` Hessian+RRHO) = {bdfe_only_sum:.4f}; "
                 f"ratio BDFE/BDE = {bdfe_only_sum/max(bde_only_sum,1e-9):.2f}.\n\n")
        fh.write("**Interpretation**: if BDFE's importance is small relative to BDE's, a "
                 "cost-aware variant dropping the 2 expensive BDFE columns (keeping only BDE) "
                 "is worth training and comparing against the full 275-feat champion — for "
                 "prospective screening of NEW molecules, BDE alone (no Hessian needed) is far "
                 "cheaper per molecule than BDFE.\n\n")
        fh.write("## Mordred re-slimming\n\n")
        fh.write(f"On top of the existing 199-feat SHAP-pruned mordred set, kept "
                 f"**{len(kept)}/{len(prunable)}** after this round's importance+correlation "
                 f"pass (dropped {len(dropped)} newly-redundant, |corr|>0.9 with a higher-ranked "
                 f"kept feature — note: importance ranking now reflects the *joint* 275-feat "
                 f"model, not the isolated mordred510 model this selection was originally made "
                 f"from, so some previously-kept feats may now look more/less useful).\n\n")
        fh.write(f"**Cost-aware candidate**: {len(cost_aware_feats)} feats (drops the 2 "
                 f"expensive BDFE cols, keeps BDE) — needs its own retrain to confirm it holds "
                 f"the accuracy; see `mordredslim271_bdegxtb_slim_selection_{TAG}.json` "
                 f"(`cost_aware_feats_total`).\n\n")
        fh.write("## Top-25 global importance\n\n| rank | feature | mean|SHAP| | family |\n|---|---|---|---|\n")
        for i, (f, v) in enumerate(mean_abs.head(25).items(), 1):
            fam = "g-xTB BDE/BDFE" if f in BDE_COLS else ("QM(72)" if f in FEATS_72 else "mordred")
            fh.write(f"| {i} | {f} | {v:.4f} | {fam} |\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
