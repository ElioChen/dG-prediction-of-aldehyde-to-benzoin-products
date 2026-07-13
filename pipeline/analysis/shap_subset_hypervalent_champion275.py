#!/usr/bin/env python
"""Tier-1a of the 2026-07-10 external-diagnosis review (REPORT_review_external_diagnosis_
20260710.md Action E): recompute SHAP restricted to the sulfonyl/P/imine/amide hard subset,
plus SHAP interaction values, instead of the existing GLOBAL top-25 (shap_slim_
mordredslim271_bdegxtb.py) which reflects the whole 220k library, not specifically the hard
tail. Answers: on just the hard subset, is the model actually trying (and failing) to isolate
the hypervalent subclass via P_int/dispersion features, or via something else entirely?

No retraining -- loads the saved champion bundle, reproduces the exact test split, tags
molecules with the same SMARTS used in deep_error_analysis_champion275.py, and computes
SHAP (XGB_d8, TreeExplainer) restricted to sulfonyl/has_P/imine/amide rows vs the rest.
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

# same SMARTS as deep_error_analysis_champion275.py (nonewg-outlier-drivers hard subset)
SMARTS = {
    "sulfonyl": "[#16X4](=[OX1])(=[OX1])",
    "has_P": "[#15]",
    "imine": "[CX3]=[NX2]",
    "amide": "[CX3](=O)[NX3]",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
HARD_TAGS = list(SMARTS.keys())


def tag(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return {k: False for k in SMARTS}
    return {k: m.HasSubstructMatch(p) for k, p in PATS.items()}


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
    print(f"bundle: {len(FEATS)} feats, test_mae={bundle['test_mae']:.3f}", flush=True)

    cons = Path(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet")
    dft = pd.read_parquet(cons, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "donor_id", "smiles", "dG_gxtb_kcal"] + PROD_QM, low_memory=False)
    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id", "smiles"] + ALD, low_memory=False).drop_duplicates("id")
    a_r = a.rename(columns={"id": "ald_id", "smiles": "ald_smiles", **{c: f"ald_{c}" for c in ALD}})
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

    prod_bde = pd.read_csv(f"{H}/products_bdfe_gxtb_descriptors.csv",
                            usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"bdfe_gxtb_kcal": "prod_bdfe_gxtb_kcal", "bde_gxtb_kcal": "prod_bde_gxtb_kcal"})
    ald_bde = pd.read_csv(f"{H}/aldehydes_bdfe_gxtb_descriptors.csv",
                           usecols=["id", "bdfe_gxtb_kcal", "bde_gxtb_kcal"]).rename(
        columns={"id": "ald_id", "bdfe_gxtb_kcal": "ald_bdfe_gxtb_kcal", "bde_gxtb_kcal": "ald_bde_gxtb_kcal"})
    for c in ["prod_bdfe_gxtb_kcal", "prod_bde_gxtb_kcal"]: prod_bde.loc[prod_bde[c].abs() > 200, c] = np.nan
    for c in ["ald_bdfe_gxtb_kcal", "ald_bde_gxtb_kcal"]: ald_bde.loc[ald_bde[c].abs() > 200, c] = np.nan

    full = p.copy(); full["ald_id"] = full["donor_id"].astype("Int64"); full = full.merge(a_r, on="ald_id", how="left")
    full = add_global(full, "smiles")
    full = full.merge(prod_mrd[["id"] + prod_mrd_cols], on="id", how="left")
    full = full.merge(ald_mrd[["ald_id"] + ald_mrd_cols], on="ald_id", how="left")
    full = full.merge(prod_bde, on="id", how="left")
    full = full.merge(ald_bde, on="ald_id", how="left")

    df = full.merge(dft, on="id").merge(cls, on="id", how="left")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS_72).reset_index(drop=True)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    print(f"labeled rows: {len(df):,}", flush=True)

    rng = np.random.default_rng(42); idx = rng.permutation(len(df))
    ntr, nva = int(.7 * len(df)), int(.9 * len(df)); te = idx[nva:]
    df_te = df.iloc[te].reset_index(drop=True)
    Xte = sc.transform(imp.transform(df_te[FEATS].values))
    Xte_df = pd.DataFrame(Xte, columns=FEATS)

    # ── tag the test set with the hard-subset SMARTS ────────────────────────
    u_ald = df_te[["ald_smiles"]].drop_duplicates()
    ald_tags = pd.DataFrame([tag(s) for s in u_ald["ald_smiles"]]).add_prefix("ald_")
    ald_tags["ald_smiles"] = u_ald["ald_smiles"].values
    df_te = df_te.merge(ald_tags, on="ald_smiles", how="left")
    u_prod = df_te[["smiles"]].drop_duplicates()
    prod_tags = pd.DataFrame([tag(s) for s in u_prod["smiles"]]).add_prefix("prod_")
    prod_tags["smiles"] = u_prod["smiles"].values
    df_te = df_te.merge(prod_tags, on="smiles", how="left")
    hard_mask = np.zeros(len(df_te), dtype=bool)
    for t in HARD_TAGS:
        hard_mask |= df_te[f"ald_{t}"].fillna(False).values | df_te[f"prod_{t}"].fillna(False).values
    print(f"hard subset (sulfonyl/P/imine/amide): {hard_mask.sum():,} / {len(df_te):,} "
          f"({hard_mask.mean()*100:.1f}%)", flush=True)

    # ── SHAP: hard subset vs background (rest of test set), same feature space ──
    rng2 = np.random.default_rng(7)
    hard_idx = np.where(hard_mask)[0]
    bg_idx_all = np.where(~hard_mask)[0]
    bg_idx = rng2.choice(bg_idx_all, size=min(len(hard_idx), len(bg_idx_all)), replace=False)
    print(f"SHAP: hard n={len(hard_idx):,}  background(matched-size) n={len(bg_idx):,}", flush=True)

    expl = shap.TreeExplainer(xgb8)
    sv_hard = expl(Xte_df.iloc[hard_idx].reset_index(drop=True))
    sv_bg = expl(Xte_df.iloc[bg_idx].reset_index(drop=True))

    mean_abs_hard = pd.Series(np.abs(sv_hard.values).mean(0), index=FEATS).sort_values(ascending=False)
    mean_abs_bg = pd.Series(np.abs(sv_bg.values).mean(0), index=FEATS)
    cmp_df = pd.DataFrame({"mean_abs_shap_hard": mean_abs_hard,
                            "mean_abs_shap_background": mean_abs_bg.reindex(mean_abs_hard.index)})
    cmp_df["hard_rank"] = range(1, len(cmp_df) + 1)
    cmp_df["background_rank"] = mean_abs_bg.rank(ascending=False).reindex(mean_abs_hard.index).astype(int)
    cmp_df["rank_shift"] = cmp_df["background_rank"] - cmp_df["hard_rank"]
    cmp_df.to_csv(OUT / f"shap_subset_hypervalent_champion275_{TAG}.csv")

    fig, ax = plt.subplots(figsize=(9, 8))
    top25 = mean_abs_hard.head(25)
    ax.barh(range(len(top25)), top25.values[::-1], color="#cb181d")
    ax.set_yticks(range(len(top25))); ax.set_yticklabels(top25.index[::-1], fontsize=8)
    ax.set_xlabel("mean |SHAP value|  (hard subset: sulfonyl/P/imine/amide)")
    ax.set_title(f"Hard-subset SHAP top-25 (n={len(hard_idx):,})\nvs global top-25 in REPORT_shap_mordredslim271_bdegxtb_20260707.md")
    savefig(f"130_shap_hard_subset_top25_{TAG}.png")

    # ── SHAP interaction values on the hard subset (capped for memory) ──────
    n_int = min(1200, len(hard_idx))
    int_idx = rng2.choice(hard_idx, size=n_int, replace=False)
    int_frame = Xte_df.iloc[int_idx].reset_index(drop=True)
    print(f"computing SHAP interaction values on {n_int} hard-subset rows...", flush=True)
    inter = expl.shap_interaction_values(int_frame)
    mean_abs_inter = np.abs(inter).mean(0)
    np.fill_diagonal(mean_abs_inter, 0.0)  # zero out main effects, keep pure interactions
    top_pair_idx = np.dstack(np.unravel_index(np.argsort(mean_abs_inter, axis=None)[::-1], mean_abs_inter.shape))[0]
    seen, pairs = set(), []
    for i, j in top_pair_idx:
        if i == j: continue
        key = tuple(sorted((int(i), int(j))))
        if key in seen: continue
        seen.add(key)
        pairs.append((FEATS[i], FEATS[j], float(mean_abs_inter[i, j])))
        if len(pairs) >= 15: break
    pairs_df = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "mean_abs_interaction"])
    pairs_df.to_csv(OUT / f"shap_interaction_top_pairs_hard_champion275_{TAG}.csv", index=False)

    top15_feat_idx = [FEATS.index(f) for f in mean_abs_hard.head(15).index]
    sub_mat = mean_abs_inter[np.ix_(top15_feat_idx, top15_feat_idx)]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sub_mat, cmap="Reds")
    ax.set_xticks(range(15)); ax.set_xticklabels([FEATS[i] for i in top15_feat_idx], rotation=90, fontsize=7)
    ax.set_yticks(range(15)); ax.set_yticklabels([FEATS[i] for i in top15_feat_idx], fontsize=7)
    ax.set_title(f"SHAP interaction strength, top-15 hard-subset features (n={n_int})")
    plt.colorbar(im, ax=ax, fraction=0.046)
    savefig(f"131_shap_interaction_heatmap_hard_{TAG}.png")

    # ── report ───────────────────────────────────────────────────────────
    rep_en = OUT / f"REPORT_shap_subset_hypervalent_champion275_{TAG}.md"
    with open(rep_en, "w") as fh:
        fh.write(f"# Subset-only SHAP on the sulfonyl/P/imine/amide hard tail ({TAG})\n\n")
        fh.write("Tier-1a of the 2026-07-10 external-diagnosis review (Action E). Same champion "
                 "bundle (`gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib`, test MAE "
                 f"{bundle['test_mae']:.3f}), no retraining. Hard subset = test-set molecules where "
                 "either the aldehyde or product SMILES matches sulfonyl/has_P/imine/amide SMARTS "
                 "(the tags found 11.2x/9.5x/3.6x/3.4x enriched in the worst-15% routed set, see "
                 f"REPORT_deep_error_analysis_champion275_20260707.md). n_hard={len(hard_idx):,} "
                 f"({hard_mask.mean()*100:.1f}% of test set), matched-size random background n={len(bg_idx):,}.\n\n")
        fh.write("## Top-25 SHAP importance, hard subset only, vs global/background rank\n\n")
        fh.write("| rank(hard) | feature | mean\\|SHAP\\|(hard) | mean\\|SHAP\\|(background) | rank(background) | rank shift |\n|---|---|---|---|---|---|\n")
        for f, r in cmp_df.head(25).iterrows():
            fh.write(f"| {int(r['hard_rank'])} | {f} | {r['mean_abs_shap_hard']:.4f} | "
                     f"{r['mean_abs_shap_background']:.4f} | {int(r['background_rank'])} | {int(r['rank_shift']):+d} |\n")
        fh.write("\nPositive rank shift = MORE important on the hard subset than on a matched-size "
                 "random background (model relies on it more/differently for hypervalent cases); "
                 "negative = less important.\n\n")
        fh.write(f"## Top-15 SHAP interaction pairs on the hard subset (n={n_int})\n\n")
        fh.write("Diagonal (main effects) excluded -- these are pure pairwise interaction strengths.\n\n")
        fh.write("| feature 1 | feature 2 | mean\\|interaction\\| |\n|---|---|---|\n")
        for _, r in pairs_df.iterrows():
            fh.write(f"| {r['feature_1']} | {r['feature_2']} | {r['mean_abs_interaction']:.4f} |\n")
        fh.write(f"\nSee `130_shap_hard_subset_top25_{TAG}.png` and `131_shap_interaction_heatmap_hard_{TAG}.png`.\n")
    print("wrote", rep_en, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
