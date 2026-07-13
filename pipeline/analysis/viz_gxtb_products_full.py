#!/usr/bin/env python
"""Deep visualization analysis of the FULL-LIBRARY g-xTB product-descriptor set.

Source: data/cross_benzoin/homo_v6/products_all.csv  (job 24128375, drained 2026-06-24 ~06:02,
concatenated 06:19). 220,724 homo-benzoin products x 72 cols: GFN2-xTB (dG_xtb_kcal) AND
g-xTB (dG_gxtb_kcal) reaction ΔG + ~50 product descriptors. 16 ADCH/QTAIM cols are empty
(submit ran MULTIWFN=0) and are reported, not plotted.

One standalone PNG per figure (no composite panels). Never overwrites: dated output dir.
Emits a markdown report with interpreted findings.
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "savefig.bbox": "tight"})

SRC = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/products_all.csv")
STAMP = "20260625"
OUT = SRC.parent / f"viz_gxtb_{STAMP}"
OUT.mkdir(exist_ok=True)
REPORT = OUT / f"REPORT_gxtb_products_full_{STAMP}.md"

DESC_ELEC = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta",
             "xtb_omega", "xtb_dipole"]
DESC_CHARGE = ["mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO", "mulliken_hydH"]
DESC_BOND = ["wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb"]
DESC_FUKUI = ["fukui_plus_ketC", "fukui_minus_ketC", "dual_ketC",
              "fukui_plus_carbC", "fukui_minus_carbC", "dual_carbC"]
DESC_STERIC = ["vbur_ketC", "vbur_carbC", "sterimol_L", "sterimol_B1", "sterimol_B5",
               "SASA_total", "P_int", "pa_ketO"]
DESC_HB = ["hb_dist", "hb_angle", "dih_core"]
USABLE = DESC_ELEC + DESC_CHARGE + DESC_BOND + DESC_FUKUI + DESC_STERIC + DESC_HB


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", p.name)
    return p


def main():
    print("loading", SRC.name)
    d = pd.read_csv(SRC, low_memory=False)
    n0 = len(d)
    empty = [c for c in d.columns if d[c].isna().all()]
    lines = []  # report

    # ---- coverage / cleaning ----
    valid = d.dropna(subset=["dG_xtb_kcal", "dG_gxtb_kcal"]).copy()
    valid["resid"] = valid["dG_gxtb_kcal"] - valid["dG_xtb_kcal"]
    # robust core for parity stats (drop pathological |ΔG|>60 from broken SCF/geom)
    core = valid[(valid["dG_xtb_kcal"].abs() < 60) & (valid["dG_gxtb_kcal"].abs() < 60)]

    lines.append(f"# Full-library g-xTB product-descriptor analysis ({STAMP})\n")
    lines.append(f"**Source**: `{SRC}` — job 24128375, drained 2026-06-24 ~06:02.\n")
    lines.append("## 1. Coverage & data quality\n")
    lines.append(f"- Rows: **{n0:,}** products (all `reaction_type=homo`, all `is_homo=True`).")
    lines.append(f"- xTB-optimized: **{int(d.xtb_optimized.sum()):,}** ({100*d.xtb_optimized.mean():.1f}%); "
                 f"error rows: {int(d.error.notna().sum()):,}.")
    lines.append(f"- Both ΔG present (GFN2 & g-xTB): **{len(valid):,}** ({100*len(valid)/n0:.1f}%).")
    lines.append(f"- Robust core |ΔG|<60 for both: **{len(core):,}** "
                 f"({100*len(core)/len(valid):.2f}% of valid); "
                 f"{len(valid)-len(core):,} pathological-|ΔG| pairs excluded from parity stats.")
    lines.append(f"- **Empty descriptor families ({len(empty)} cols, MULTIWFN=0)**: "
                 f"all ADCH + QTAIM are 100% NaN -> unusable here:\n  `{', '.join(empty)}`\n")

    # ---- Fig: descriptor-family completeness ----
    fam = {"electronic": DESC_ELEC, "mulliken": DESC_CHARGE, "WBO": DESC_BOND,
           "Fukui": DESC_FUKUI, "steric": DESC_STERIC, "H-bond": DESC_HB,
           "ADCH(empty)": [c for c in empty if c.startswith("adch")],
           "QTAIM(empty)": [c for c in empty if c.startswith("qtaim")]}
    cov = {k: 100 * (1 - d[v].isna().mean().mean()) for k, v in fam.items() if v}
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(cov.keys())
    ax.barh(ks, [cov[k] for k in ks],
            color=["#2b8cbe" if cov[k] > 50 else "#cb181d" for k in ks])
    ax.set_xlabel("mean non-null coverage (%)")
    ax.set_title("Descriptor-family completeness (full library)")
    ax.set_xlim(0, 100)
    save(fig, "01_descriptor_family_coverage.png")

    # ---- Fig: ΔG distributions GFN2 vs g-xTB ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-40, 40, 120)
    ax.hist(core["dG_xtb_kcal"], bins=bins, alpha=0.55, label=f"GFN2-xTB (μ={core.dG_xtb_kcal.mean():.2f})",
            color="#4575b4")
    ax.hist(core["dG_gxtb_kcal"], bins=bins, alpha=0.55, label=f"g-xTB (μ={core.dG_gxtb_kcal.mean():.2f})",
            color="#d73027")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("ΔG_homo-benzoin (kcal/mol)")
    ax.set_ylabel("count")
    ax.set_title("Reaction ΔG distribution: GFN2-xTB vs g-xTB")
    ax.legend()
    save(fig, "02_dG_distributions.png")

    # ---- Fig: parity hexbin ----
    fig, ax = plt.subplots(figsize=(6, 5.5))
    hb = ax.hexbin(core["dG_xtb_kcal"], core["dG_gxtb_kcal"], gridsize=70,
                   bins="log", cmap="viridis", mincnt=1)
    lim = [-40, 40]
    ax.plot(lim, lim, "w--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    r = np.corrcoef(core["dG_xtb_kcal"], core["dG_gxtb_kcal"])[0, 1]
    mae = core["resid"].abs().mean(); bias = core["resid"].mean()
    rmse = np.sqrt((core["resid"] ** 2).mean())
    ax.set_xlabel("ΔG GFN2-xTB (kcal/mol)")
    ax.set_ylabel("ΔG g-xTB (kcal/mol)")
    ax.set_title(f"g-xTB vs GFN2 parity  (r={r:.3f}, bias={bias:+.2f}, MAE={mae:.2f}, RMSE={rmse:.2f})")
    fig.colorbar(hb, ax=ax, label="log10 count")
    save(fig, "03_parity_gxtb_vs_gfn2.png")

    # ---- Fig: residual distribution ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(core["resid"], bins=np.linspace(-30, 30, 120), color="#756bb1")
    ax.axvline(bias, color="r", ls="--", label=f"mean {bias:+.2f}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("g-xTB − GFN2  ΔG residual (kcal/mol)")
    ax.set_ylabel("count")
    ax.set_title("Method disagreement (g-xTB minus GFN2)")
    ax.legend()
    save(fig, "04_residual_distribution.png")

    lines.append("## 2. g-xTB vs GFN2-xTB agreement (the baseline question)\n")
    lines.append(f"- Pearson r = **{r:.3f}**, bias (g-xTB−GFN2) = **{bias:+.2f}**, "
                 f"MAE = **{mae:.2f}**, RMSE = **{rmse:.2f}** kcal/mol (core, n={len(core):,}).")
    lines.append(f"- GFN2 mean ΔG **{core.dG_xtb_kcal.mean():.2f}** (exergonic) vs g-xTB **{core.dG_gxtb_kcal.mean():.2f}** "
                 f"-> the two methods disagree on reaction *sign* on average: a **{bias:+.2f} kcal/mol systematic shift**, "
                 f"not just scatter.\n")

    # ---- residual vs descriptors: what drives disagreement ----
    corrs = {}
    for c in USABLE:
        if c in core and core[c].notna().sum() > 1000:
            corrs[c] = core[["resid", c]].corr().iloc[0, 1]
    cs = pd.Series(corrs).dropna().sort_values(key=lambda s: s.abs(), ascending=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    top = cs.head(18)[::-1]
    ax.barh(top.index, top.values,
            color=["#d73027" if v > 0 else "#4575b4" for v in top.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pearson r  (descriptor vs g-xTB−GFN2 residual)")
    ax.set_title("What drives g-xTB / GFN2 disagreement?")
    save(fig, "05_residual_descriptor_correlations.png")

    # ---- residual vs top driver scatter (hexbin) ----
    drv = cs.index[0]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sub = core[[drv, "resid"]].dropna()
    ax.hexbin(sub[drv], sub["resid"], gridsize=70, bins="log", cmap="magma", mincnt=1)
    ax.axhline(0, color="w", lw=0.8, ls="--")
    ax.set_xlabel(drv)
    ax.set_ylabel("g-xTB − GFN2 residual (kcal/mol)")
    ax.set_title(f"Top disagreement driver: {drv}  (r={cs.iloc[0]:+.3f})")
    save(fig, "06_residual_vs_top_driver.png")

    lines.append("## 3. What drives the disagreement\n")
    lines.append("Top |r| descriptors vs (g-xTB−GFN2) residual:\n")
    for c, v in cs.head(8).items():
        lines.append(f"- `{c}`: r = {v:+.3f}")
    lines.append("")

    # ---- ΔG drivers (g-xTB) ----
    gcorr = {}
    for c in USABLE:
        if c in core and core[c].notna().sum() > 1000:
            gcorr[c] = core[["dG_gxtb_kcal", c]].corr().iloc[0, 1]
    gs = pd.Series(gcorr).dropna().sort_values(key=lambda s: s.abs(), ascending=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    top = gs.head(18)[::-1]
    ax.barh(top.index, top.values,
            color=["#d73027" if v > 0 else "#4575b4" for v in top.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pearson r  (descriptor vs g-xTB ΔG)")
    ax.set_title("Descriptor → g-xTB ΔG correlations")
    save(fig, "07_dG_descriptor_correlations.png")

    # ---- correlation heatmap among key descriptors + ΔG ----
    keep = ["dG_gxtb_kcal", "dG_xtb_kcal"] + [c for c in USABLE
            if c in core and core[c].notna().sum() > 1000]
    cm = core[keep].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep))); ax.set_xticklabels(keep, rotation=90, fontsize=7)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep, fontsize=7)
    ax.set_title("Descriptor correlation matrix (full library, core)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "08_correlation_heatmap.png")

    # ---- electronic: HOMO-LUMO gap & electrophilicity ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(core["xtb_gap"].dropna(), bins=120, color="#1c9099")
    ax.set_xlabel("xtb HOMO-LUMO gap (eV)")
    ax.set_ylabel("count"); ax.set_title("Electronic: HOMO–LUMO gap distribution")
    save(fig, "09_homolumo_gap.png")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sub = core[["xtb_omega", "dG_gxtb_kcal"]].dropna()
    sub = sub[sub.xtb_omega.between(sub.xtb_omega.quantile(.001), sub.xtb_omega.quantile(.999))]
    ax.hexbin(sub.xtb_omega, sub.dG_gxtb_kcal, gridsize=70, bins="log", cmap="cividis", mincnt=1)
    ax.set_xlabel("xtb global electrophilicity ω")
    ax.set_ylabel("g-xTB ΔG (kcal/mol)")
    ax.set_title("Electrophilicity ω vs g-xTB ΔG")
    save(fig, "10_omega_vs_dG.png")

    # ---- Fukui dual at ketone carbon ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    v = core["dual_ketC"].dropna()
    v = v[v.between(v.quantile(.005), v.quantile(.995))]
    ax.hist(v, bins=120, color="#dd1c77")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("dual descriptor at ketone C (electrophilic site)")
    ax.set_ylabel("count"); ax.set_title("Fukui dual descriptor @ ketone carbon")
    save(fig, "11_fukui_dual_ketC.png")

    # ---- steric: buried volume ket vs carb ----
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sub = core[["vbur_ketC", "vbur_carbC"]].dropna()
    ax.hexbin(sub.vbur_ketC, sub.vbur_carbC, gridsize=60, bins="log", cmap="YlGnBu", mincnt=1)
    ax.set_xlabel("%V_bur ketone C"); ax.set_ylabel("%V_bur carbinol C")
    ax.set_title("Steric: buried volume, ketone vs carbinol C")
    save(fig, "12_vbur_ket_vs_carb.png")

    # ---- H-bond geometry in product ----
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sub = core[["hb_dist", "hb_angle"]].dropna()
    sub = sub[(sub.hb_dist.between(1.2, 4.0)) & (sub.hb_angle.between(80, 180))]
    ax.hexbin(sub.hb_dist, sub.hb_angle, gridsize=70, bins="log", cmap="inferno", mincnt=1)
    ax.set_xlabel("intramolecular H-bond distance (Å)")
    ax.set_ylabel("H-bond angle (deg)")
    ax.set_title("Product α-hydroxyketone H-bond geometry")
    save(fig, "13_hbond_geometry.png")

    # ---- outlier tail characterization ----
    out = valid[(valid.dG_xtb_kcal.abs() > 60) | (valid.dG_gxtb_kcal.abs() > 60)]
    lines.append("## 4. Pathological ΔG tail\n")
    lines.append(f"- **{len(out):,}** pairs have |ΔG|>60 in at least one method "
                 f"({100*len(out)/len(valid):.2f}%) — likely broken SCF / strained geometry, "
                 "candidates to route to DFT (never delete; see no-ΔG-extreme-filtering).")
    if len(out):
        lines.append(f"- of those, g-xTB-only extreme: {int((out.dG_gxtb_kcal.abs()>60).sum() & ~(out.dG_xtb_kcal.abs()>60)).__index__() if False else ((out.dG_gxtb_kcal.abs()>60)&(out.dG_xtb_kcal.abs()<=60)).sum():,}; "
                     f"GFN2-only extreme: {((out.dG_xtb_kcal.abs()>60)&(out.dG_gxtb_kcal.abs()<=60)).sum():,}; both: {((out.dG_xtb_kcal.abs()>60)&(out.dG_gxtb_kcal.abs()>60)).sum():,}.")
    lines.append("")
    lines.append("## 5. Figures\n")
    for p in sorted(OUT.glob("*.png")):
        lines.append(f"- `{p.name}`")
    lines.append("")
    lines.append("## 6. Takeaways\n")
    lines.append(f"1. **g-xTB is not a drop-in for GFN2 on ΔG**: r≈{r:.2f} but a **{bias:+.1f} kcal/mol systematic bias** "
                 "flips the average reaction from exergonic (GFN2) to endergonic (g-xTB). Calibrate before use as a Δ-baseline.")
    lines.append("2. The disagreement is **structured, not random** — it correlates with the descriptors in fig 05, "
                 "so a small linear/affine correction keyed on those should collapse most of the bias.")
    lines.append("3. **ADCH+QTAIM are absent full-library** (MULTIWFN=0). If those features matter for the surrogate, "
                 "they must be back-filled on a sampled subset (see multiwfn-env-and-screen-gap).")
    REPORT.write_text("\n".join(lines))
    print("report ->", REPORT)
    print("done. figures in", OUT)


if __name__ == "__main__":
    main()
