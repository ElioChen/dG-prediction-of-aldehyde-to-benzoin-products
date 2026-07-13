#!/usr/bin/env python
"""Deep, multi-dimensional analysis of the noise band / high-error molecules for the current
champion model (MORDREDSLIM271_BDEGXTB, test MAE 1.503, job 24468737).

Complements the earlier SHAP-attribution-space clustering done for baseline_72
(shap_baseline72.py, 53_shap_fingerprint_clusters.png -- 4 heterogeneous attribution-pattern
clusters) with a STRUCTURE-space view: does the worst-error tail correspond to a handful of
recognizable functional-group/chemotype clusters, or is it genuinely heterogeneous? Also
checks whether the tail composition changed now that BDE/BDFE features are in the model
(previously identified drivers: EWG/sulfonyl-F, soft-S/P/B/Si, imine -- see
nonewg-outlier-drivers / screen-v6-funcgroup-analysis memories) and quantifies how much of
the remaining error is "real" (>> noise floor) vs within the established ~1.57+/-0.013 noise
band (data-limited, per descriptor-search-exhausted memory).

Reuses test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv (already has per-molecule error,
uncertainty, route_to_dft, cls, BDE/BDFE values -- no need to reload the full feature matrix).
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem, Draw
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d")
NOISE_MEAN, NOISE_STD = 1.571, 0.013  # baseline_72 robustness study, REPORT_robustness_baseline72_20260702.md

# functional-group SMARTS, extending the screen_v6 EWG set with the previously-identified
# non-EWG drivers (soft-S, P, B, Si, imine) per nonewg-outlier-drivers memory
SMARTS = {
    "sulfonyl": "[#16X4](=[OX1])(=[OX1])",
    "sulfonyl_F": "[#16X4](=[OX1])(=[OX1])F",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile": "[CX2]#[NX1]",
    "imine": "[CX3]=[NX2]",
    "has_B": "[#5]",
    "has_Si": "[#14]",
    "has_P": "[#15]",
    "soft_S_thioether": "[#16X2]",
    "furan_O": "[o]",
    "thiophene_S": "[s]",
    "azine_N": "[n]",
    "halogen": "[F,Cl,Br,I]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[CX3](=O)[NX3]",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}


def tag(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return {k: False for k in SMARTS}
    return {k: m.HasSubstructMatch(p) for k, p in PATS.items()}


def savefig(name):
    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", name, flush=True)


def main():
    te = pd.read_csv(f"{H}/viz_gxtb_20260625/test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv")
    n = len(te)
    print(f"test set: {n:,} molecules, MAE={te['error'].mean():.3f}", flush=True)

    # ── 0. noise-band accounting: how much of the tail is genuinely "real" error? ──
    noise_hi = NOISE_MEAN + 3 * NOISE_STD  # ~3 sigma of the established seed-reshuffle noise band
    frac_within = float((te["error"] <= noise_hi).mean())
    frac_real_tail = 1 - frac_within
    print(f"noise band 3-sigma cutoff = {noise_hi:.3f} kcal/mol; "
          f"{frac_within*100:.1f}% of test set within noise, {frac_real_tail*100:.1f}% genuinely elevated", flush=True)

    # ── 1. functional-group tags on aldehyde + product SMILES ──────────────
    u_ald = te[["ald_smiles"]].drop_duplicates()
    ald_tags = pd.DataFrame([tag(s) for s in u_ald["ald_smiles"]]).add_prefix("ald_")
    ald_tags["ald_smiles"] = u_ald["ald_smiles"].values
    te = te.merge(ald_tags, on="ald_smiles", how="left")
    u_prod = te[["smiles"]].drop_duplicates()
    prod_tags = pd.DataFrame([tag(s) for s in u_prod["smiles"]]).add_prefix("prod_")
    prod_tags["smiles"] = u_prod["smiles"].values
    te = te.merge(prod_tags, on="smiles", how="left")
    tag_cols = [c for c in te.columns if c.startswith("ald_") or c.startswith("prod_")]
    tag_cols = [c for c in tag_cols if te[c].dtype == bool]

    routed = te[te["route_to_dft"] == True].copy()
    background = te[te["route_to_dft"] == False].copy()
    print(f"routed (worst 15%): {len(routed):,}  background: {len(background):,}", flush=True)

    enrich = []
    for c in tag_cols:
        p_routed = routed[c].mean(); p_bg = background[c].mean()
        ratio = p_routed / p_bg if p_bg > 0 else float("inf") if p_routed > 0 else 1.0
        enrich.append({"tag": c, "pct_routed": p_routed * 100, "pct_background": p_bg * 100,
                        "enrichment_ratio": ratio, "n_routed_with_tag": int(routed[c].sum())})
    enrich_df = pd.DataFrame(enrich).sort_values("enrichment_ratio", ascending=False)
    enrich_df.to_csv(OUT / f"error_tag_enrichment_champion275_{TAG}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = enrich_df[enrich_df["n_routed_with_tag"] >= 20].head(15)
    ax.barh(top["tag"][::-1], top["enrichment_ratio"][::-1], color="#cb181d")
    ax.axvline(1, color="k", lw=1, ls="--")
    ax.set_xlabel("enrichment ratio (routed / background), n>=20 in routed")
    ax.set_title(f"champion275 worst-15% functional-group enrichment (n={len(routed):,})")
    savefig(f"113_error_tag_enrichment_{TAG}.png")

    # ── 2. structure-space clustering of the worst-15% (Morgan FP + KMeans) ─
    fps = []
    valid_idx = []
    for i, s in enumerate(routed["smiles"]):
        m = Chem.MolFromSmiles(str(s))
        if m is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=512)
        fps.append(np.array(fp)); valid_idx.append(i)
    X = np.array(fps)
    routed_v = routed.iloc[valid_idx].reset_index(drop=True)
    n_clusters = min(6, max(2, len(routed_v) // 500))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X)
    routed_v["struct_cluster"] = km.labels_
    routed_v[["id", "ald_smiles", "smiles", "cls", "error", "struct_cluster"] + tag_cols].to_csv(
        OUT / f"struct_clusters_champion275_{TAG}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    prof = routed_v.groupby("struct_cluster")["error"].agg(["mean", "count"])
    ax.bar(prof.index.astype(str), prof["mean"], color="#2171b5")
    for i, (m_, c_) in enumerate(zip(prof["mean"], prof["count"])):
        ax.text(i, m_, f"n={c_}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("structure cluster"); ax.set_ylabel("mean |error| (kcal/mol)")
    ax.set_title(f"champion275 worst-15% structure clusters (Morgan FP, k={n_clusters})")
    savefig(f"114_struct_cluster_error_{TAG}.png")

    # which tags dominate each cluster?
    cluster_profile = []
    for c in range(n_clusters):
        sub = routed_v[routed_v["struct_cluster"] == c]
        row = {"cluster": c, "n": len(sub), "mean_error": sub["error"].mean()}
        for t in tag_cols:
            row[t] = sub[t].mean()
        cluster_profile.append(row)
    cluster_profile_df = pd.DataFrame(cluster_profile)
    cluster_profile_df.to_csv(OUT / f"struct_cluster_profile_champion275_{TAG}.csv", index=False)

    # ── 3. error vs molecular size / uncertainty / BDE-BDFE magnitude ──────
    te["MolWt"] = te["smiles"].apply(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else np.nan)
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, c in [("aromatic", "#2171b5"), ("aliphatic", "#cb181d")]:
        mk = te["cls"] == s
        ax.scatter(te.loc[mk, "MolWt"], te.loc[mk, "error"], s=4, alpha=0.3, color=c, label=s)
    ax.axhline(noise_hi, color="k", ls="--", lw=1, label=f"3-sigma noise cutoff ({noise_hi:.2f})")
    ax.set_xlabel("product MolWt (g/mol)"); ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title("champion275 error vs size, scope-split"); ax.legend()
    savefig(f"115_error_vs_molwt_scope_{TAG}.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(te["uncertainty_pi_width"], te["error"], s=4, alpha=0.25, color="#238b45")
    ax.axhline(noise_hi, color="k", ls="--", lw=1)
    ax.set_xlabel("quantile PI width (uncertainty)"); ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title("champion275 error vs model uncertainty (does UQ know what it doesn't know?)")
    savefig(f"116_error_vs_uncertainty_{TAG}.png")

    # ── 4. worst-30 molecule grid, annotated with functional-group tags ────
    worst30 = te.sort_values("error", ascending=False).head(30).reset_index(drop=True)
    mols, legends = [], []
    for _, r in worst30.iterrows():
        m = Chem.MolFromSmiles(str(r["ald_smiles"]))
        if m is None: continue
        active_tags = [t.replace("ald_", "") for t in tag_cols if t.startswith("ald_") and r.get(t)]
        mols.append(m)
        legends.append(f"id={r['id']} err={r['error']:.2f}\n{r['cls']} {','.join(active_tags[:3])}")
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(260, 220), legends=legends)
        img.save(OUT / f"117_worst30_tagged_champion275_{TAG}.png")
        print(f"wrote 117_worst30_tagged_champion275_{TAG}.png", flush=True)

    # ── report ───────────────────────────────────────────────────────────
    rep = OUT / f"REPORT_deep_error_analysis_champion275_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# Deep error / noise-band analysis: MORDREDSLIM271_BDEGXTB ({TAG})\n\n")
        fh.write(f"Test set n={n:,}, overall MAE={te['error'].mean():.3f}. Noise-band reference "
                 f"(baseline_72, 5 reshuffled seeds): {NOISE_MEAN} +/- {NOISE_STD} kcal/mol "
                 f"(REPORT_robustness_baseline72_20260702.md).\n\n")
        fh.write(f"## 1. How much of the tail is real vs noise?\n\n")
        fh.write(f"Using a 3-sigma cutoff on the noise band ({noise_hi:.3f} kcal/mol): "
                 f"**{frac_within*100:.1f}%** of test molecules have error within what pure label "
                 f"noise could explain; **{frac_real_tail*100:.1f}%** ({int(frac_real_tail*n):,} "
                 f"molecules) show genuinely elevated error the model is failing to capture, not "
                 f"just label jitter.\n\n")
        fh.write(f"## 2. Functional-group enrichment in the worst 15% (routed) set\n\n")
        fh.write("Ratio = P(tag | routed) / P(tag | background); >1 means over-represented in "
                 "hard cases. Filtered to tags with >=20 occurrences in the routed set.\n\n")
        fh.write("| tag | % in routed | % in background | enrichment | n(routed) |\n|---|---|---|---|---|\n")
        for _, r in top.iterrows():
            fh.write(f"| {r['tag']} | {r['pct_routed']:.1f}% | {r['pct_background']:.1f}% | "
                     f"{r['enrichment_ratio']:.2f}x | {int(r['n_routed_with_tag'])} |\n")
        fh.write(f"\n## 3. Structure-space clustering (Morgan FP, k={n_clusters})\n\n")
        fh.write("Unlike the earlier SHAP-attribution-space clustering (baseline_72, 4 "
                 "heterogeneous attribution patterns), this clusters by actual molecular "
                 "structure to see if error concentrates on recognizable chemotypes.\n\n")
        fh.write("| cluster | n | mean error | dominant tags (>50% prevalence) |\n|---|---|---|---|\n")
        for _, r in cluster_profile_df.iterrows():
            dom = [t for t in tag_cols if r[t] > 0.5]
            fh.write(f"| {int(r['cluster'])} | {int(r['n'])} | {r['mean_error']:.2f} | {', '.join(dom) or '(none dominant)'} |\n")
        fh.write(f"\n## 4. Scope split (aromatic vs aliphatic) and size dependence\n\n")
        for s in ["aromatic", "aliphatic"]:
            mk = te["cls"] == s
            fh.write(f"- **{s}**: n={mk.sum():,}, MAE={te.loc[mk,'error'].mean():.3f}\n")
        fh.write(f"\nSee `115_error_vs_molwt_scope_{TAG}.png` for the full size x scope error plot, "
                 f"`116_error_vs_uncertainty_{TAG}.png` for whether the uncertainty-routing signal "
                 f"actually tracks true error, and `117_worst30_tagged_champion275_{TAG}.png` for "
                 f"the worst-30 molecule structures annotated with functional-group tags.\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
