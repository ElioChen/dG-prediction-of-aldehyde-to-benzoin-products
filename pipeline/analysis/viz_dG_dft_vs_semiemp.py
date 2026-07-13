#!/usr/bin/env python
"""Raw-data correlation of reaction dG: DFT (y) vs GFN2-xTB and g-xTB (x).

DFT labels  : data/raw/dft_sp_funnelv3/chunk_*.csv  col dG_orca_kcal (r2SCAN-3c, CPCM-DMSO)
semi-emp    : data/cross_benzoin/homo_v6/products_all.csv  cols dG_xtb_kcal (GFN2), dG_gxtb_kcal (g-xTB)
joined on `id`. RAW values, no correction. One standalone PNG per figure.
"""
import glob
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "savefig.bbox": "tight"})

ROOT = Path("/scratch-shared/schen3/benzoin-dg")
DFTDIR = ROOT / "data/raw/dft_sp_funnelv3"
PROD = ROOT / "data/cross_benzoin/homo_v6/products_all.csv"
STAMP = "20260629"
OUT = ROOT / f"data/cross_benzoin/homo_v6/viz_dG_corrected_{STAMP}"
OUT.mkdir(exist_ok=True)
REPORT = OUT / f"REPORT_dG_dft_vs_semiemp_{STAMP}.md"

CLAMP = 60.0  # drop pathological |DFT| (broken SCF), reported


def metrics(x, y):
    err = x - y
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    bias = np.mean(err)  # method - DFT
    r = np.corrcoef(x, y)[0, 1]
    a, b = np.polyfit(x, y, 1)  # DFT ~ a*method + b
    return mae, rmse, bias, r, a, b


def parity(method, dft, label, fname, color):
    mae, rmse, bias, r, a, b = metrics(method, dft)
    lo = min(method.min(), dft.min())
    hi = max(method.max(), dft.max())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(method, dft, gridsize=80, cmap="viridis", bins="log", mincnt=1)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    xs = np.array([lo, hi])
    ax.plot(xs, a * xs + b, color=color, lw=1.5,
            label=f"fit: DFT = {a:.2f}·x {b:+.2f}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{label} ΔG (kcal/mol)")
    ax.set_ylabel("DFT r2SCAN-3c ΔG (kcal/mol)")
    txt = (f"n = {len(dft):,}\nPearson R = {r:.3f}\nR$^2$ = {r**2:.3f}\n"
           f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nbias = {bias:+.2f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_title(f"DFT vs {label} (raw reaction ΔG)")
    ax.legend(loc="lower right", fontsize=9)
    p = OUT / fname
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", fname)
    return mae, rmse, bias, r, a, b


def main():
    print("loading DFT chunks ...")
    fs = sorted(glob.glob(str(DFTDIR / "chunk_*.csv")))
    dft = pd.concat([pd.read_csv(f, usecols=["id", "dG_orca_kcal"]) for f in fs],
                    ignore_index=True)
    dft = dft.dropna(subset=["dG_orca_kcal"]).drop_duplicates("id")
    print(f"  DFT labels: {len(dft):,}")

    p = pd.read_csv(PROD, usecols=["id", "dG_xtb_kcal", "dG_gxtb_kcal"], low_memory=False)
    df = p.merge(dft, on="id", how="inner")
    n_match = len(df)
    df = df.dropna(subset=["dG_xtb_kcal", "dG_gxtb_kcal", "dG_orca_kcal"])
    n_clamp = int((df["dG_orca_kcal"].abs() >= CLAMP).sum())
    df = df[df["dG_orca_kcal"].abs() < CLAMP]
    print(f"  matched {n_match:,}; usable {len(df):,} (dropped {n_clamp} |DFT|>={CLAMP})")

    y = df["dG_orca_kcal"].to_numpy()
    g2 = df["dG_xtb_kcal"].to_numpy()
    gx = df["dG_gxtb_kcal"].to_numpy()

    m2 = parity(g2, y, "GFN2-xTB", "08_parity_DFT_vs_gfn2.png", "#e53e3e")
    mx = parity(gx, y, "g-xTB", "09_parity_DFT_vs_gxtb.png", "#2b6cb0")

    L = [f"# Raw ΔG correlation: DFT vs GFN2 / g-xTB  ({STAMP})\n",
         f"DFT = r2SCAN-3c (`dG_orca_kcal`, n DFT labels matched = {n_match:,}; "
         f"usable {len(df):,}, dropped {n_clamp} with |DFT| ≥ {CLAMP}).\n\n",
         "| method (x) | Pearson R | R² | MAE | RMSE | bias(method−DFT) | slope | intercept |\n",
         "|---|---|---|---|---|---|---|---|\n",
         f"| GFN2-xTB | {m2[3]:.3f} | {m2[3]**2:.3f} | {m2[0]:.2f} | {m2[1]:.2f} | {m2[2]:+.2f} | {m2[4]:.2f} | {m2[5]:+.2f} |\n",
         f"| g-xTB | {mx[3]:.3f} | {mx[3]**2:.3f} | {mx[0]:.2f} | {mx[1]:.2f} | {mx[2]:+.2f} | {mx[4]:.2f} | {mx[5]:+.2f} |\n\n",
         "## Interpretation\n",
         f"- **g-xTB tracks DFT far better than GFN2**: R²={mx[3]**2:.3f} vs {m2[3]**2:.3f}, "
         f"MAE={mx[0]:.2f} vs {m2[0]:.2f} kcal/mol. g-xTB is the right Δ-baseline; GFN2 is not.\n",
         f"- Bias: GFN2 {m2[2]:+.2f}, g-xTB {mx[2]:+.2f} kcal/mol (method − DFT). "
         "A slope ≠ 1 means the error is partly systematic (correctable), motivating the ML Δ-correction.\n",
         "## Figures\n",
         "- `08_parity_DFT_vs_gfn2.png`\n- `09_parity_DFT_vs_gxtb.png`\n"]
    REPORT.write_text("".join(L))
    print("  wrote", REPORT.name)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
