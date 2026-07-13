#!/usr/bin/env python
"""A/B route analysis for the fully-solvated g-xTB pilot campaign (1% = 2191 mols).

Option A = funnel_v3 (GFN2/CREST) geometry, SPs: GFN2-ALPB / g-xTB-cosmo / DFT-CPCM(r2SCAN-3c).
Option B = g-xTB --opt --cosmo dmso geometry, SPs: g-xTB-cosmo / DFT-CPCM.
A is the trusted reference; B is exploratory (does opt-in-solvent help or hurt?).

Questions answered:
  1. On A geometry, does g-xTB-cosmo ΔG track DFT-CPCM? vs the GFN2-ALPB baseline.
  2. Does B geometry improve g-xTB->DFT agreement, and how sensitive is the DFT reference
     itself to the geometry change (DFT_B vs DFT_A)?
  3. g-xTB opt convergence + geometry blow-up (RMSD) diagnostics.

Outputs: standalone PNGs (one figure per file) + a deep markdown report. Prior outputs
are never overwritten (dated filenames).
"""
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("data/raw/screen_v6/dft_sp_r2scan3c/gxtb_solv_pilot")
OUT = R / "ab_analysis"
OUT.mkdir(exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M")


def err_stats(pred, ref):
    """Return dict of error metrics for pred vs ref (both kcal/mol), NaNs dropped pairwise."""
    m = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[m], ref[m]
    d = p - r
    n = len(d)
    if n < 2:
        return dict(n=n, mae=np.nan, rmse=np.nan, bias=np.nan, r2=np.nan,
                    med=np.nan, p90=np.nan, max=np.nan)
    ss_res = np.sum((r - p) ** 2)
    ss_tot = np.sum((r - r.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(n=n, mae=np.mean(np.abs(d)), rmse=np.sqrt(np.mean(d ** 2)),
                bias=np.mean(d), r2=r2, med=np.median(np.abs(d)),
                p90=np.percentile(np.abs(d), 90), max=np.max(np.abs(d)))


def scatter(pred, ref, title, xlabel, ylabel, fname, stats):
    m = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[m], ref[m]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(r, p, s=8, alpha=0.35, edgecolors="none")
    lo = min(r.min(), p.min())
    hi = max(r.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    txt = (f"n={stats['n']}\nMAE={stats['mae']:.2f}\nRMSE={stats['rmse']:.2f}\n"
           f"bias={stats['bias']:+.2f}\nR²={stats['r2']:.3f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def hist_err(d, title, xlabel, fname):
    d = d[np.isfinite(d)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(d, bins=60, alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.mean(d), color="r", lw=1.2, ls="--",
               label=f"mean {np.mean(d):+.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def main():
    files = sorted(R.glob("chunk_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_all = len(df)

    # Clean: drop rows whose note flags an error, and coerce ΔG cols to numeric.
    cols = ["dG_gfn2_alpb_A", "dG_gxtb_cosmo_A", "dG_dft_cpcm_A",
            "dG_gxtb_cosmo_B", "dG_dft_cpcm_B"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    note = df.get("note", pd.Series([""] * len(df))).astype(str)
    err_mask = note.str.contains("err|timeout|fail", case=False, na=False)
    n_err = int(err_mask.sum())
    clean = df[~err_mask].copy()

    gfn2A = clean["dG_gfn2_alpb_A"].to_numpy(float)
    gxtbA = clean["dG_gxtb_cosmo_A"].to_numpy(float)
    dftA = clean["dG_dft_cpcm_A"].to_numpy(float)
    gxtbB = clean["dG_gxtb_cosmo_B"].to_numpy(float)
    dftB = clean["dG_dft_cpcm_B"].to_numpy(float)

    # Core comparisons
    s_gfn2A = err_stats(gfn2A, dftA)          # GFN2-ALPB baseline vs DFT (route A)
    s_gxtbA = err_stats(gxtbA, dftA)          # g-xTB-cosmo vs DFT (route A) <-- headline
    s_gxtbB = err_stats(gxtbB, dftB)          # g-xTB-cosmo vs DFT (route B)
    s_dftBvA = err_stats(dftB, dftA)          # DFT geometry sensitivity (B geom vs A geom)
    # g-xTB self-consistency across geometries (does opt-in-solvent move g-xTB ΔG?)
    s_gxtbBvA = err_stats(gxtbB, gxtbA)

    # Option B convergence + blow-up diagnostics
    def conv_frac(col):
        v = clean.get(col)
        if v is None:
            return np.nan, 0
        vb = v.astype(str).str.lower().isin(["true", "1", "1.0"])
        return vb.mean(), int((~vb).sum())
    conv_ald, nfail_ald = conv_frac("gxtbopt_conv_ald")
    conv_bz, nfail_bz = conv_frac("gxtbopt_conv_bz")
    rmsd_ald = pd.to_numeric(clean.get("rmsd_ald"), errors="coerce")
    rmsd_bz = pd.to_numeric(clean.get("rmsd_bz"), errors="coerce")

    # Figures (one per file)
    scatter(gxtbA, dftA, "g-xTB-cosmo vs DFT-CPCM  (Route A / funnel geom)",
            "ΔG DFT-CPCM r2SCAN-3c (kcal/mol)", "ΔG g-xTB-cosmo (kcal/mol)",
            f"scatter_gxtb_vs_dft_routeA_{STAMP}.png", s_gxtbA)
    scatter(gfn2A, dftA, "GFN2-ALPB vs DFT-CPCM  (Route A / funnel geom)",
            "ΔG DFT-CPCM r2SCAN-3c (kcal/mol)", "ΔG GFN2-ALPB (kcal/mol)",
            f"scatter_gfn2_vs_dft_routeA_{STAMP}.png", s_gfn2A)
    scatter(gxtbB, dftB, "g-xTB-cosmo vs DFT-CPCM  (Route B / g-xTB opt-in-solvent geom)",
            "ΔG DFT-CPCM r2SCAN-3c (kcal/mol)", "ΔG g-xTB-cosmo (kcal/mol)",
            f"scatter_gxtb_vs_dft_routeB_{STAMP}.png", s_gxtbB)
    scatter(dftB, dftA, "DFT-CPCM geometry sensitivity  (B geom vs A geom)",
            "ΔG DFT-CPCM on funnel geom A (kcal/mol)",
            "ΔG DFT-CPCM on g-xTB-opt geom B (kcal/mol)",
            f"scatter_dft_B_vs_A_{STAMP}.png", s_dftBvA)
    hist_err(gxtbA - dftA, "g-xTB-cosmo − DFT-CPCM error (Route A)",
             "ΔΔG (kcal/mol)", f"hist_gxtb_err_routeA_{STAMP}.png")
    hist_err(dftB - dftA, "DFT-CPCM(B) − DFT-CPCM(A) geometry shift",
             "ΔΔG (kcal/mol)", f"hist_dft_geomshift_{STAMP}.png")
    rr = rmsd_bz[np.isfinite(rmsd_bz)]
    if len(rr):
        hist_err(rr.to_numpy(), "g-xTB opt heavy-atom RMSD vs funnel geom (benzoin)",
                 "coarse RMSD (Å)", f"hist_rmsd_bz_optB_{STAMP}.png")

    # Report
    def row(name, s):
        return (f"| {name} | {s['n']} | {s['mae']:.2f} | {s['rmse']:.2f} | "
                f"{s['bias']:+.2f} | {s['r2']:.3f} | {s['med']:.2f} | "
                f"{s['p90']:.2f} | {s['max']:.1f} |")

    rep = OUT / f"REPORT_gxtb_solv_ab_{STAMP}.md"
    with open(rep, "w") as f:
        f.write(f"# g-xTB fully-solvated A/B route analysis — 1% pilot\n\n")
        f.write(f"_Generated {datetime.now():%Y-%m-%d %H:%M}. "
                f"Source: `{R}/chunk_*.csv` ({len(files)} chunks)._\n\n")
        f.write("## Dataset\n\n")
        f.write(f"- Rows total: **{n_all}**  |  err-flagged dropped: **{n_err}**  "
                f"|  clean rows analysed: **{len(clean)}**\n")
        f.write("- Reaction ΔG: 2 R-CHO → benzoin, fully solvated (DMSO). "
                "DFT-CPCM r2SCAN-3c is ground truth.\n")
        f.write("- **Route A** = funnel_v3 (GFN2/CREST) geometry, single-points on it. "
                "**Route B** = g-xTB `--opt --cosmo dmso` geometry, single-points on it.\n\n")

        f.write("## Headline error metrics (vs DFT-CPCM, kcal/mol)\n\n")
        f.write("| comparison | n | MAE | RMSE | bias | R² | median|Δ| | p90|Δ| | max|Δ| |\n")
        f.write("|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        f.write(row("GFN2-ALPB vs DFT  (Route A baseline)", s_gfn2A) + "\n")
        f.write(row("**g-xTB-cosmo vs DFT  (Route A)**", s_gxtbA) + "\n")
        f.write(row("g-xTB-cosmo vs DFT  (Route B)", s_gxtbB) + "\n")
        f.write(row("DFT(B geom) vs DFT(A geom)", s_dftBvA) + "\n")
        f.write(row("g-xTB(B) vs g-xTB(A)", s_gxtbBvA) + "\n\n")

        f.write("## Interpretation\n\n")
        # 1. g-xTB vs GFN2 on the trusted geometry
        better = "better" if s_gxtbA["mae"] < s_gfn2A["mae"] else "worse"
        f.write(f"**1. Does g-xTB-cosmo beat GFN2-ALPB as a solvated DFT surrogate (Route A)?** "
                f"g-xTB MAE {s_gxtbA['mae']:.2f} vs GFN2 {s_gfn2A['mae']:.2f} kcal/mol — "
                f"g-xTB is **{better}**. Bias: g-xTB {s_gxtbA['bias']:+.2f}, "
                f"GFN2 {s_gfn2A['bias']:+.2f}. R²: {s_gxtbA['r2']:.3f} vs {s_gfn2A['r2']:.3f}. "
                f"Tail: g-xTB p90|Δ|={s_gxtbA['p90']:.2f}, max={s_gxtbA['max']:.1f} kcal/mol.\n\n")
        # 2. Route B effect
        dmae = s_gxtbB["mae"] - s_gxtbA["mae"]
        verdict = ("does NOT help (worse)" if dmae > 0.05 else
                   "helps" if dmae < -0.05 else "is a wash")
        f.write(f"**2. Does optimising the geometry in solvent (Route B) help?** "
                f"g-xTB→DFT MAE moves {dmae:+.2f} kcal/mol (A {s_gxtbA['mae']:.2f} → "
                f"B {s_gxtbB['mae']:.2f}); opt-in-solvent **{verdict}**. "
                f"The DFT reference itself shifts under the geometry change: "
                f"DFT(B) vs DFT(A) MAE={s_dftBvA['mae']:.2f}, max={s_dftBvA['max']:.1f} kcal/mol "
                f"— i.e. the g-xTB opt geometry moves the *true* ΔG by this much, so Route B "
                f"is only trustworthy where that shift is small.\n\n")
        # 3. convergence
        f.write(f"**3. Route B robustness.** g-xTB opt convergence: aldehyde "
                f"{conv_ald*100:.1f}% ({nfail_ald} non-converged), benzoin "
                f"{conv_bz*100:.1f}% ({nfail_bz} non-converged). ")
        if np.isfinite(rmsd_bz).any():
            f.write(f"Heavy-atom RMSD (benzoin, g-xTB-opt vs funnel): "
                    f"median {np.nanmedian(rmsd_bz):.2f} Å, "
                    f"p90 {np.nanpercentile(rmsd_bz,90):.2f} Å, "
                    f"max {np.nanmax(rmsd_bz):.2f} Å — large values flag opt blow-ups.\n\n")
        else:
            f.write("\n\n")
        f.write("## Figures\n\n")
        for p in sorted(OUT.glob(f"*_{STAMP}.png")):
            f.write(f"- `{p.name}`\n")

    print(f"clean rows: {len(clean)} (dropped {n_err} err)")
    print(f"g-xTB-cosmo vs DFT  RouteA: MAE={s_gxtbA['mae']:.2f} R2={s_gxtbA['r2']:.3f} bias={s_gxtbA['bias']:+.2f}")
    print(f"GFN2-ALPB   vs DFT  RouteA: MAE={s_gfn2A['mae']:.2f} R2={s_gfn2A['r2']:.3f}")
    print(f"g-xTB-cosmo vs DFT  RouteB: MAE={s_gxtbB['mae']:.2f} R2={s_gxtbB['r2']:.3f}")
    print(f"DFT(B) vs DFT(A) geom sens: MAE={s_dftBvA['mae']:.2f} max={s_dftBvA['max']:.1f}")
    print(f"report: {rep}")


if __name__ == "__main__":
    main()
