#!/usr/bin/env python
"""Redundancy diagnostic: how correlated is aldehyde C-H BDFE with the existing 271-feat
mordredslim271 descriptor set (72 QM + 199 SHAP-pruned mordred)? If BDFE is highly
correlated with something already in the model, its ML value should be mostly redundant;
if it's weakly correlated with everything, it's carrying genuinely new information -- this
is the piece needed to interpret the earlier finding (BDFE only weakly correlates with dG
itself, r~0.04-0.10, yet explicit raw-E BDE gave a real +0.024 MAE gain on top of
mordredslim271, see bde-descriptor-idea memory).
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d")

ALD = ["xtb_HOMO","xtb_LUMO","xtb_gap","xtb_IP","xtb_EA","xtb_mu","xtb_eta","xtb_omega","xtb_dipole",
  "mulliken_CHO_C","mulliken_CHO_O","fukui_plus_CHO_C","fukui_minus_CHO_C","dual_descriptor_CHO_C",
  "wbo_CO","pa_CHO_O","vbur_CHO_C","sterimol_L","sterimol_B1","sterimol_B5","SASA_total","P_int"]


def savefig(name):
    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", name, flush=True)


def main():
    bdfe = pd.read_csv(f"{H}/aldehydes_bdfe2_descriptors.csv").rename(columns={"id": "ald_id"})
    bdfe["ald_id"] = bdfe["ald_id"].astype(float).astype("Int64")
    bdfe = bdfe.dropna(subset=["bdfe_xtb_kcal"])
    bdfe = bdfe[bdfe["bdfe_xtb_kcal"].abs() <= 200]  # drop the 5 pathological SCF-failure rows

    a = pd.read_csv(f"{H}/aldehydes_all.csv", usecols=["id"] + ALD, low_memory=False).rename(columns={"id": "ald_id"})
    a["ald_id"] = a["ald_id"].astype(float).astype("Int64")

    kept_mordred = json.load(open(f"{H}/viz_gxtb_20260625/mordred_slim_selection_20260703.json"))["kept_mordred"]
    ald_mordred_names = [c[len("ald_mordred_"):] for c in kept_mordred if c.startswith("ald_mordred_")]
    ald_mrd_header = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", nrows=0).columns
    ald_mrd_want = ["id"] + [c for c in ald_mrd_header if c.replace("mordred_", "") in ald_mordred_names]
    ald_mrd = pd.read_csv(f"{H}/aldehydes_mordred_descriptors.csv", usecols=ald_mrd_want, low_memory=False)
    ald_mrd = ald_mrd.rename(columns={"id": "ald_id"})
    ald_mrd["ald_id"] = ald_mrd["ald_id"].astype(float).astype("Int64")
    mrd_cols = [c for c in ald_mrd.columns if c.startswith("mordred_")]
    for c in mrd_cols:
        ald_mrd[c] = pd.to_numeric(ald_mrd[c], errors="coerce")

    df = bdfe.merge(a, on="ald_id", how="inner").merge(ald_mrd, on="ald_id", how="left")
    print(f"rows: {len(df):,}", flush=True)

    feat_cols = ALD + mrd_cols
    corrs = []
    for c in feat_cols:
        sub = df[["bdfe_xtb_kcal", c]].dropna()
        if len(sub) < 1000 or sub[c].std() == 0:
            continue
        r, p = stats.pearsonr(sub["bdfe_xtb_kcal"], sub[c])
        corrs.append({"feature": c, "r": r, "abs_r": abs(r), "p": p, "n": len(sub),
                      "family": "aldehyde QM (72-feat)" if c in ALD else "mordred (199-feat)"})
    corr_df = pd.DataFrame(corrs).sort_values("abs_r", ascending=False).reset_index(drop=True)
    corr_df.to_csv(OUT / f"bdfe_feature_correlation_{TAG}.csv", index=False)
    print(f"computed correlation vs {len(corr_df)} existing features", flush=True)
    print(corr_df.head(15).to_string(), flush=True)

    top20 = corr_df.head(20)
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#cb181d" if f == "aldehyde QM (72-feat)" else "#2171b5" for f in top20["family"][::-1]]
    ax.barh(range(len(top20)), top20["r"].values[::-1], color=colors)
    ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20["feature"].values[::-1], fontsize=8)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Pearson r with aldehyde C-H BDFE")
    ax.set_title(f"top-20 existing features most correlated with BDFE (n={len(df):,})\n"
                f"red=aldehyde QM(72), blue=mordred(199)")
    savefig(f"105_bdfe_top_correlated_feats_{TAG}.png")

    # scatter for the single most-correlated feature, for a visual sanity check
    top1 = corr_df.iloc[0]
    sub = df[["bdfe_xtb_kcal", top1["feature"]]].dropna()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sub[top1["feature"]], sub["bdfe_xtb_kcal"], s=3, alpha=0.15, color="#2171b5")
    ax.set_xlabel(top1["feature"]); ax.set_ylabel("aldehyde C-H BDFE (kcal/mol)")
    ax.set_title(f"BDFE vs its single most-correlated existing feature\nr={top1['r']:.3f} (n={len(sub):,})")
    savefig(f"106_bdfe_vs_top_feature_{TAG}.png")

    rep = OUT / f"REPORT_bdfe_feature_redundancy_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# BDFE redundancy vs the existing mordredslim271 feature set ({TAG})\n\n")
        fh.write(f"n={len(df):,} molecules. Correlated aldehyde C-H BDFE against all "
                f"{len(feat_cols)} available existing features (22 aldehyde-QM + "
                f"{len(mrd_cols)} mordred kept-set feats).\n\n")
        fh.write("## Top-15 most correlated existing features\n\n")
        fh.write("| rank | feature | family | Pearson r | n |\n|---|---|---|---|---|\n")
        for i, row in corr_df.head(15).iterrows():
            fh.write(f"| {i+1} | {row['feature']} | {row['family']} | {row['r']:.3f} | {row['n']:,} |\n")
        max_abs_r = corr_df["abs_r"].max()
        n_over_03 = (corr_df["abs_r"] > 0.3).sum()
        n_over_05 = (corr_df["abs_r"] > 0.5).sum()
        fh.write(f"\n## Summary\n\n")
        fh.write(f"- Strongest single correlation: |r|={max_abs_r:.3f} ({corr_df.iloc[0]['feature']})\n")
        fh.write(f"- {n_over_03}/{len(corr_df)} existing features have |r|>0.3 with BDFE\n")
        fh.write(f"- {n_over_05}/{len(corr_df)} existing features have |r|>0.5 with BDFE\n\n")
        verdict = ("BDFE is NOT strongly redundant with any single existing feature"
                  if max_abs_r < 0.5 else
                  "BDFE overlaps substantially with at least one existing feature")
        fh.write(f"**{verdict}.** Combined with its weak direct correlation with dG "
                f"(r~0.04-0.10, see REPORT_aldehyde_bdfe_analysis_{TAG}.md) but real "
                f"tree-model MAE gain (+0.024) when added explicitly, the most consistent "
                f"interpretation is that BDFE carries **distributed, nonlinear/interaction "
                f"information** that individual existing descriptors don't capture on "
                f"their own -- plausible given BDFE is a genuinely different physical "
                f"quantity (a THERMODYNAMIC, ohess-derived free energy of a bond-breaking "
                f"process) vs. the existing descriptors (static single-point electronic/"
                f"steric properties of the intact molecule).\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
