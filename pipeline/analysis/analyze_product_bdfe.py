#!/usr/bin/env python
"""Deep-dive analysis: product new-C-C bond dissociation free energy (BDFE) distribution,
and its correlation with the reaction dG (both g-xTB and real DFT). Mirrors
analyze_aldehyde_bdfe.py exactly but for the product side (products_bdfe2_descriptors.csv).
Standalone figures, one per file.
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d")


def savefig(name):
    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", name, flush=True)


def main():
    bdfe = pd.read_csv(f"{H}/products_bdfe2_descriptors.csv")

    # products_all.csv has ~1008 duplicate-id rows (pre-existing data-quality wrinkle,
    # unrelated to BDFE) -- dedupe here to avoid inflating merged row counts
    p = pd.read_csv(f"{H}/products_all.csv", usecols=["id", "dG_gxtb_kcal"], low_memory=False).drop_duplicates("id")

    cons = Path(f"{R}/data/raw/dft_sp_funnelv3/dft_labels_all.parquet")
    dft = pd.read_parquet(cons, columns=["id", "dG_orca_kcal"]).dropna(subset=["dG_orca_kcal"]).drop_duplicates("id", keep="last")

    cls = pd.read_parquet(f"{H}/aldehyde_class.parquet")

    df = p.merge(bdfe[["id", "bde_E_kcal", "bdfe_xtb_kcal"]], on="id", how="inner")
    df = df.merge(dft, on="id", how="left")
    df = df.merge(cls, on="id", how="left")
    df = df.dropna(subset=["bdfe_xtb_kcal"]).reset_index(drop=True)
    print(f"rows with BDFE: {len(df):,}; of those with DFT dG: {df['dG_orca_kcal'].notna().sum():,}", flush=True)

    # sanity filter, same pattern as the aldehyde-side analysis
    n_before = len(df)
    outliers = df[df["bdfe_xtb_kcal"].abs() > 200]
    if len(outliers):
        outliers[["id", "bde_E_kcal", "bdfe_xtb_kcal"]].to_csv(
            OUT / f"prod_bdfe_pathological_outliers_{TAG}.csv", index=False)
    df = df[df["bdfe_xtb_kcal"].abs() <= 200].reset_index(drop=True)
    print(f"sanity filter |BDFE|<=200 kcal/mol: dropped {n_before - len(df)}/{n_before} "
         f"pathological rows (saved to prod_bdfe_pathological_outliers_{TAG}.csv)", flush=True)

    # ── 1) distribution ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["bdfe_xtb_kcal"], bins=100, color="#cb181d", edgecolor="none")
    ax.axvline(df["bdfe_xtb_kcal"].median(), color="k", ls="--", lw=1,
              label=f"median={df['bdfe_xtb_kcal'].median():.1f}")
    ax.set_xlabel("product new C-C BDFE, DMSO (kcal/mol)"); ax.set_ylabel("count")
    ax.set_title(f"product C-C BDFE distribution (n={len(df):,})\n"
                f"mean={df['bdfe_xtb_kcal'].mean():.1f} std={df['bdfe_xtb_kcal'].std():.1f}")
    ax.legend()
    savefig(f"120_prod_bdfe_distribution_{TAG}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for s, c in [("aromatic", "#2171b5"), ("aliphatic", "#cb181d")]:
        sub = df[df["cls"] == s]["bdfe_xtb_kcal"]
        ax.hist(sub, bins=80, alpha=0.5, color=c, label=f"{s} (n={len(sub):,}, mean={sub.mean():.1f})", density=True)
    ax.set_xlabel("product C-C BDFE (kcal/mol)"); ax.set_ylabel("density")
    ax.set_title("product C-C BDFE distribution by scope")
    ax.legend()
    savefig(f"121_prod_bdfe_by_scope_{TAG}.png")

    # ── 2) correlation with dG (gxtb, full coverage) ─────────────────────
    sub = df.dropna(subset=["dG_gxtb_kcal"])
    r_gxtb, p_gxtb = stats.pearsonr(sub["bdfe_xtb_kcal"], sub["dG_gxtb_kcal"])
    rho_gxtb, _ = stats.spearmanr(sub["bdfe_xtb_kcal"], sub["dG_gxtb_kcal"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sub["bdfe_xtb_kcal"], sub["dG_gxtb_kcal"], s=3, alpha=0.15, color="#cb181d")
    ax.set_xlabel("product C-C BDFE (kcal/mol)"); ax.set_ylabel("dG_gxtb (kcal/mol)")
    ax.set_title(f"product BDFE vs g-xTB reaction dG (n={len(sub):,})\n"
                f"Pearson r={r_gxtb:.3f} (p={p_gxtb:.1e})  Spearman rho={rho_gxtb:.3f}")
    savefig(f"122_prod_bdfe_vs_dG_gxtb_{TAG}.png")

    # ── 3) correlation with dG (real DFT, smaller coverage) ──────────────
    sub2 = df.dropna(subset=["dG_orca_kcal"])
    if len(sub2) > 50:
        r_dft, p_dft = stats.pearsonr(sub2["bdfe_xtb_kcal"], sub2["dG_orca_kcal"])
        rho_dft, _ = stats.spearmanr(sub2["bdfe_xtb_kcal"], sub2["dG_orca_kcal"])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub2["bdfe_xtb_kcal"], sub2["dG_orca_kcal"], s=3, alpha=0.2, color="#238b45")
        ax.set_xlabel("product C-C BDFE (kcal/mol)"); ax.set_ylabel("dG_orca DFT (kcal/mol)")
        ax.set_title(f"product BDFE vs real DFT reaction dG (n={len(sub2):,})\n"
                    f"Pearson r={r_dft:.3f} (p={p_dft:.1e})  Spearman rho={rho_dft:.3f}")
        savefig(f"123_prod_bdfe_vs_dG_dft_{TAG}.png")
    else:
        r_dft = rho_dft = None
        print("too few DFT-labeled rows for a meaningful DFT correlation plot", flush=True)

    # ── 4) BDFE vs error of the g-xTB baseline ────────────────────────────
    sub3 = df.dropna(subset=["dG_orca_kcal", "dG_gxtb_kcal"]).copy()
    sub3["gxtb_err"] = (sub3["dG_gxtb_kcal"] - sub3["dG_orca_kcal"]).abs()
    if len(sub3) > 50:
        r_err, p_err = stats.pearsonr(sub3["bdfe_xtb_kcal"], sub3["gxtb_err"])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub3["bdfe_xtb_kcal"], sub3["gxtb_err"], s=3, alpha=0.2, color="#6a51a3")
        ax.set_xlabel("product C-C BDFE (kcal/mol)"); ax.set_ylabel("|g-xTB - DFT| error (kcal/mol)")
        ax.set_title(f"product BDFE vs g-xTB baseline error (n={len(sub3):,})\nPearson r={r_err:.3f} (p={p_err:.1e})")
        savefig(f"124_prod_bdfe_vs_gxtb_error_{TAG}.png")
    else:
        r_err = None

    # ── 5) reactant vs product BDFE side by side (direct comparison) ─────
    ald_bdfe = pd.read_csv(f"{H}/aldehydes_bdfe2_descriptors.csv").rename(columns={"id": "ald_id"})
    ald_bdfe = ald_bdfe[ald_bdfe["bdfe_xtb_kcal"].abs() <= 200]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ald_bdfe["bdfe_xtb_kcal"], bins=100, alpha=0.5, color="#2171b5",
           label=f"aldehyde C-H (n={len(ald_bdfe):,}, mean={ald_bdfe['bdfe_xtb_kcal'].mean():.1f})", density=True)
    ax.hist(df["bdfe_xtb_kcal"], bins=100, alpha=0.5, color="#cb181d",
           label=f"product C-C (n={len(df):,}, mean={df['bdfe_xtb_kcal'].mean():.1f})", density=True)
    ax.set_xlabel("BDFE (kcal/mol)"); ax.set_ylabel("density")
    ax.set_title("aldehyde C-H vs product C-C BDFE, side by side")
    ax.legend()
    savefig(f"125_ald_vs_prod_bdfe_{TAG}.png")

    rep = OUT / f"REPORT_product_bdfe_analysis_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# Product C-C BDFE: distribution + dG correlation ({TAG})\n\n")
        fh.write(f"n={len(df):,} molecules with valid, physically-sane BDFE (of {len(bdfe):,} total "
                f"in the sidecar, {bdfe['bdfe_xtb_kcal'].notna().mean()*100:.1f}% overall fill rate; "
                f"{len(outliers)} pathological outliers with |BDFE|>200 kcal/mol excluded).\n\n")
        fh.write("## Distribution\n\n")
        fh.write(f"- mean={df['bdfe_xtb_kcal'].mean():.2f}, median={df['bdfe_xtb_kcal'].median():.2f}, "
                f"std={df['bdfe_xtb_kcal'].std():.2f} kcal/mol\n")
        fh.write(f"- range: {df['bdfe_xtb_kcal'].min():.1f} to {df['bdfe_xtb_kcal'].max():.1f} kcal/mol\n")
        for s in ["aromatic", "aliphatic"]:
            sub_s = df[df["cls"] == s]["bdfe_xtb_kcal"]
            if len(sub_s) > 50:
                fh.write(f"- **{s}** (n={len(sub_s):,}): mean={sub_s.mean():.2f}, std={sub_s.std():.2f}\n")
        fh.write(f"\n**Compared to aldehyde C-H BDFE** (n={len(ald_bdfe):,}): mean="
                f"{ald_bdfe['bdfe_xtb_kcal'].mean():.2f}, std={ald_bdfe['bdfe_xtb_kcal'].std():.2f} "
                f"kcal/mol -- product C-C bond is "
                f"{'WEAKER' if df['bdfe_xtb_kcal'].mean() < ald_bdfe['bdfe_xtb_kcal'].mean() else 'STRONGER'} "
                f"on average (delta={df['bdfe_xtb_kcal'].mean()-ald_bdfe['bdfe_xtb_kcal'].mean():.1f} kcal/mol).\n")
        fh.write("\n## Correlation with reaction dG\n\n")
        fh.write(f"- vs **dG_gxtb** (n={len(sub):,}): Pearson r={r_gxtb:.3f} (p={p_gxtb:.2e}), "
                f"Spearman rho={rho_gxtb:.3f}\n")
        if r_dft is not None:
            fh.write(f"- vs **dG_orca (real DFT)** (n={len(sub2):,}): Pearson r={r_dft:.3f} "
                    f"(p={p_dft:.2e}), Spearman rho={rho_dft:.3f}\n")
        if r_err is not None:
            fh.write(f"\n## Correlation with g-xTB baseline error\n\n")
            fh.write(f"- BDFE vs |g-xTB - DFT| (n={len(sub3):,}): Pearson r={r_err:.3f} (p={p_err:.2e})\n")
        fh.write("\n## Interpretation\n\n")
        corr_strength = "weak" if abs(r_gxtb) < 0.3 else ("moderate" if abs(r_gxtb) < 0.6 else "strong")
        fh.write(f"Product C-C BDFE shows a **{corr_strength}** linear correlation with the reaction dG "
                f"(r={r_gxtb:.3f} vs g-xTB) -- consistent with the aldehyde-side finding. Note this "
                f"bond is mechanistically the MOST relevant one (it's the bond literally formed in "
                f"the reaction), yet the raw correlation is still weak, which fits the earlier "
                f"finding that BDFE (either side) only gave a noise-level MAE gain "
                f"(see REPORT_bdfe_full_augment_20260704.md and bde-descriptor-idea memory) when "
                f"added explicitly to the correction model. Whether descriptors can PREDICT this "
                f"BDFE well (a separate, nonlinear-reconstruction question) is answered by the "
                f"surrogate model (see REPORT_bde_surrogate_*.md, prod_bdfe target).\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
