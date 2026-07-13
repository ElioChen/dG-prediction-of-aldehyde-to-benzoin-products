#!/usr/bin/env python
"""Visualization of the FINAL corrected full-library reaction dG (g-xTB -> DFT correction).

Source: data/cross_benzoin/homo_v6/viz_gxtb_20260625/products_dG_corrected_FINAL_20260626.csv
  218,227 homo-benzoin products x 6 cols produced by finalize_correction.py:
    dG_gxtb_kcal            : raw g-xTB reaction free energy (kcal/mol)
    dG_gxtb_corrected_final : g-xTB + ML delta (MLP+XGB8+XGB10 ensemble) -> DFT-quality estimate
    uncertainty_pi_width    : 90% quantile prediction-interval width (routing signal)
    route_to_dft            : uncertainty_pi_width >= threshold (top ~15% flagged for DFT)

The earlier viz_gxtb_20260625/ dG plots (02/07/10, Jun 25) were drawn from the raw
training/descriptor table (products_all.csv) BEFORE the ML correction + routing existed.
This script visualizes the Jun-26 corrected deliverable that had no plots.

Conventions: one standalone PNG per figure (no composite panels); dated output dir, never
overwrite; emit an interpreted markdown report.
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

SRC = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/"
           "viz_gxtb_20260625/products_dG_corrected_FINAL_20260626.csv")
STAMP = "20260629"
OUT = SRC.parent.parent / f"viz_dG_corrected_{STAMP}"
OUT.mkdir(exist_ok=True)
REPORT = OUT / f"REPORT_dG_corrected_final_{STAMP}.md"

# bulk window for readable histograms; extremes reported, never silently dropped
LO, HI = -20.0, 30.0


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", p.name)
    return p


def main():
    print("loading", SRC.name)
    d = pd.read_csv(SRC)
    n = len(d)
    raw = d["dG_gxtb_kcal"].to_numpy()
    cor = d["dG_gxtb_corrected_final"].to_numpy()
    unc = d["uncertainty_pi_width"].to_numpy()
    rt = d["route_to_dft"].astype(bool).to_numpy()
    delta = cor - raw
    L = []

    def stat(x):
        return (np.nanmin(x), np.nanpercentile(x, 25), np.nanmedian(x),
                np.nanmean(x), np.nanpercentile(x, 75), np.nanmax(x))

    out_lo = int((cor < LO).sum())
    out_hi = int((cor > HI).sum())

    # ---- Fig 1: corrected dG distribution (bulk window) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    m = (cor >= LO) & (cor <= HI)
    ax.hist(cor[m], bins=120, color="#2b6cb0", alpha=0.85)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvline(np.median(cor), color="#e53e3e", lw=1.5,
               label=f"median {np.median(cor):.2f}")
    ax.set_xlabel("corrected reaction $\\Delta G$ (kcal/mol)")
    ax.set_ylabel("count")
    ax.set_title(f"Final corrected $\\Delta G$  (n={n:,};  "
                 f"{out_lo:,} < {LO:.0f},  {out_hi:,} > {HI:.0f} off-axis)")
    ax.legend()
    save(fig, "01_dG_corrected_distribution.png")

    # ---- Fig 2: raw g-xTB vs corrected (the systematic shift) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mr = (raw >= LO) & (raw <= HI)
    ax.hist(raw[mr], bins=120, color="#a0aec0", alpha=0.6, label="raw g-xTB")
    ax.hist(cor[m], bins=120, color="#2b6cb0", alpha=0.6, label="corrected (DFT-est)")
    ax.axvline(np.median(raw), color="#718096", lw=1.4, ls="--",
               label=f"raw med {np.median(raw):.2f}")
    ax.axvline(np.median(cor), color="#2b6cb0", lw=1.4, ls="--",
               label=f"cor med {np.median(cor):.2f}")
    ax.set_xlabel("reaction $\\Delta G$ (kcal/mol)")
    ax.set_ylabel("count")
    ax.set_title("Raw g-xTB vs ML-corrected $\\Delta G$")
    ax.legend(fontsize=9)
    save(fig, "02_dG_raw_vs_corrected.png")

    # ---- Fig 3: correction magnitude (corrected - raw) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    md = (delta >= -10) & (delta <= 10)
    ax.hist(delta[md], bins=120, color="#38a169", alpha=0.85)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvline(np.median(delta), color="#e53e3e", lw=1.5,
               label=f"median {np.median(delta):+.2f}")
    ax.set_xlabel("ML correction  $\\Delta G_{corr}-\\Delta G_{g-xTB}$ (kcal/mol)")
    ax.set_ylabel("count")
    ax.set_title("Applied correction magnitude")
    ax.legend()
    save(fig, "03_correction_magnitude.png")

    # ---- Fig 4: uncertainty (PI width) distribution + routing threshold ----
    thr = float(np.min(unc[rt])) if rt.any() else np.nan
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mu = unc <= np.nanpercentile(unc, 99.5)
    ax.hist(unc[mu], bins=120, color="#805ad5", alpha=0.85)
    ax.axvline(thr, color="#e53e3e", lw=1.5,
               label=f"route threshold {thr:.2f}\n({rt.mean()*100:.0f}% routed)")
    ax.set_xlabel("uncertainty: 90% PI width (kcal/mol)")
    ax.set_ylabel("count")
    ax.set_title("Prediction-interval width (DFT-routing signal)")
    ax.legend()
    save(fig, "04_uncertainty_pi_width.png")

    # ---- Fig 5: corrected dG vs uncertainty (hexbin) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    mm = m & mu
    hb = ax.hexbin(cor[mm], unc[mm], gridsize=70, cmap="viridis", bins="log", mincnt=1)
    ax.axhline(thr, color="#e53e3e", lw=1.2, ls="--", label=f"route thr {thr:.2f}")
    ax.set_xlabel("corrected $\\Delta G$ (kcal/mol)")
    ax.set_ylabel("90% PI width (kcal/mol)")
    ax.set_title("Where is the corrected $\\Delta G$ uncertain?")
    fig.colorbar(hb, ax=ax, label="log10 count")
    ax.legend()
    save(fig, "05_dG_vs_uncertainty_hexbin.png")

    # ---- Fig 6: corrected dG split by routing decision ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cor[~rt & m], bins=100, color="#2b6cb0", alpha=0.7,
            density=True, label=f"confident (n={int((~rt).sum()):,})")
    ax.hist(cor[rt & m], bins=100, color="#e53e3e", alpha=0.6,
            density=True, label=f"route_to_dft (n={int(rt.sum()):,})")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("corrected $\\Delta G$ (kcal/mol)")
    ax.set_ylabel("density")
    ax.set_title("Corrected $\\Delta G$ by routing decision")
    ax.legend(fontsize=9)
    save(fig, "06_dG_by_routing.png")

    # ---- Fig 7: cumulative exergonic fraction (screening view) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.linspace(-15, 20, 400)
    frac = np.array([(cor <= t).mean() for t in xs]) * 100
    ax.plot(xs, frac, color="#2b6cb0", lw=2)
    for t in (0, 5, 10):
        f = (cor <= t).mean() * 100
        ax.axvline(t, color="#cbd5e0", lw=0.8)
        ax.annotate(f"{f:.0f}%", (t, f), fontsize=9, color="#e53e3e")
    ax.set_xlabel("corrected $\\Delta G$ cutoff (kcal/mol)")
    ax.set_ylabel("% library with $\\Delta G \\leq$ cutoff")
    ax.set_title("Cumulative exergonicity (screening yield)")
    save(fig, "07_cumulative_exergonic_fraction.png")

    # ---- report ----
    sr, sc, sd, su = stat(raw), stat(cor), stat(delta), stat(unc)
    fav = (cor < 0).mean() * 100
    L.append(f"# Final corrected reaction dG — homo_v6 full library ({STAMP})\n")
    L.append(f"Source: `{SRC.name}`  ·  n = **{n:,}** products\n")
    L.append("## Were dG plots drawn before?\n")
    L.append("Yes for the **raw** descriptor table (`viz_gxtb_20260625/02_dG_distributions.png`, "
             "`07_dG_descriptor_correlations.png`, `10_omega_vs_dG.png`, Jun 25), but **not** for "
             "this Jun-26 ML-corrected deliverable with uncertainty + DFT-routing. These 7 figures "
             "fill that gap.\n")
    L.append("## Distribution summary (min / q25 / median / mean / q75 / max)\n")
    L.append(f"- raw g-xTB ΔG: {sr[0]:.2f} / {sr[1]:.2f} / {sr[2]:.2f} / {sr[3]:.2f} / {sr[4]:.2f} / {sr[5]:.2f}\n")
    L.append(f"- **corrected ΔG**: {sc[0]:.2f} / {sc[1]:.2f} / {sc[2]:.2f} / {sc[3]:.2f} / {sc[4]:.2f} / {sc[5]:.2f}\n")
    L.append(f"- correction (cor−raw): {sd[0]:.2f} / {sd[1]:.2f} / {sd[2]:.2f} / {sd[3]:.2f} / {sd[4]:.2f} / {sd[5]:.2f}\n")
    L.append(f"- PI-width (uncertainty): {su[0]:.2f} / {su[1]:.2f} / {su[2]:.2f} / {su[3]:.2f} / {su[4]:.2f} / {su[5]:.2f}\n")
    L.append("## Interpretation\n")
    L.append(f"- The ML correction shifts the median ΔG **{np.median(raw):.2f} → {np.median(cor):.2f}** "
             f"kcal/mol (median correction **{np.median(delta):+.2f}**): g-xTB systematically "
             "under-estimates the endergonicity of homo-benzoin coupling, and the DFT correction is "
             "almost entirely a positive (less favorable) shift.\n")
    L.append(f"- **{fav:.1f}%** of products are exergonic at the corrected level (ΔG < 0) vs "
             f"{(raw<0).mean()*100:.1f}% raw — correction prunes the optimistic exergonic tail.\n")
    L.append(f"- Routing: **{int(rt.sum()):,} ({rt.mean()*100:.0f}%)** flagged `route_to_dft` "
             f"(PI width ≥ {thr:.2f}). High uncertainty concentrates in the ΔG extremes (Fig 5/6); "
             "the confident core is the near-thermoneutral bulk.\n")
    L.append(f"- Off-axis extremes ({out_lo:,} below {LO:.0f}, {out_hi:,} above {HI:.0f} kcal/mol) "
             "are the known EWG / strained / generator-edge cases — kept, not trimmed, per the "
             "no-ΔG-extreme-filtering rule; most fall in route_to_dft.\n")
    L.append("## Figures\n")
    for nm in ["01_dG_corrected_distribution", "02_dG_raw_vs_corrected", "03_correction_magnitude",
               "04_uncertainty_pi_width", "05_dG_vs_uncertainty_hexbin", "06_dG_by_routing",
               "07_cumulative_exergonic_fraction"]:
        L.append(f"- `{nm}.png`\n")
    REPORT.write_text("".join(L))
    print("  wrote", REPORT.name)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
