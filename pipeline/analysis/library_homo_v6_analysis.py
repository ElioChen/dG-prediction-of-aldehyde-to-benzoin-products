#!/usr/bin/env python
"""Cross-benzoin homo library (homo_v6) — completeness + ΔG screen analysis.

Reads the concatenated full-library featurization (products_all.csv, ~220k homo
benzoin products) and characterizes:
  * featurization success / per-column completeness (Multiwfn cols empty BY DESIGN),
  * the xTB and g-xTB ΔG distributions and exergonic ("favorable", ΔG<0) fractions,
  * g-xTB vs xTB method agreement (bias / MAE / correlation),

Standalone figures (one per file, per no-composite-figures preference) go to
data/analysis/library_homo_v6/<ts>_*.png and a markdown report alongside.
"""
from __future__ import annotations
import datetime, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/scratch-shared/schen3/benzoin-dg"
OUT = f"{REPO}/data/cross_benzoin/homo_v6"
ANA = f"{REPO}/data/analysis/library_homo_v6"
os.makedirs(ANA, exist_ok=True)
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BLUE, RED = "#3182bd", "#d7301f"


def _hist(series, title, xlabel, fname, color=BLUE):
    s = series.dropna()
    plt.figure(figsize=(7.4, 5.0))
    plt.hist(s, bins=80, color=color, alpha=0.85)
    plt.axvline(0, color="k", lw=1, ls="--")
    neg = 100 * (s < 0).mean()
    plt.title(f"{title}\nn={len(s):,}  median={s.median():.1f}  favorable(<0)={neg:.1f}%")
    plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(f"{ANA}/{TS}_{fname}", dpi=130); plt.close()
    return len(s), float(s.median()), float(neg)


def main() -> int:
    df = pd.read_csv(f"{OUT}/products_all.csv", low_memory=False)
    n = len(df)
    lines = [f"# Cross-benzoin homo library (homo_v6) — analysis", "",
             f"_Generated {TS} from `products_all.csv` (n={n:,})._", ""]

    # ---- completeness ----
    err = df["error"].notna().sum() if "error" in df else 0
    ok = n - err
    nulls = df.isna().mean().sort_values(ascending=False)
    design_empty = [c for c in df.columns if c.startswith(("adch_", "qtaim_"))]
    real_cols = [c for c in df.columns if c not in design_empty]
    real_null = df[real_cols].isna().mean()
    worst = real_null[real_null > 0].sort_values(ascending=False).head(12)
    lines += ["## 1. Completeness",
              f"- Rows (homo products): **{n:,}**",
              f"- Rows with an `error`: **{err:,}** → molecule success rate **{100*ok/n:.2f}%**",
              f"- Multiwfn cols empty *by design* (MULTIWFN=0 at full scale): {len(design_empty)} "
              f"(`adch_*`, `qtaim_*`)",
              "- Real columns with residual nulls (calc failures):"]
    for c, v in worst.items():
        lines.append(f"  - `{c}`: {v*100:.2f}%")
    lines.append("")

    # ---- ΔG distributions ----
    nx, mx, fx = _hist(df.get("dG_xtb_kcal"), "Homo benzoin ΔG (GFN2-xTB)",
                       "ΔG_xTB (kcal/mol)", "dG_xtb_hist.png", BLUE)
    lines += ["## 2. ΔG screen (xTB)",
              f"- Valid `dG_xtb_kcal`: **{nx:,}** ({100*nx/n:.1f}%)",
              f"- Median ΔG_xTB = **{mx:.2f}** kcal/mol; exergonic (ΔG<0) = **{fx:.1f}%**",
              f"- Figure: `{TS}_dG_xtb_hist.png`", ""]

    if "dG_gxtb_kcal" in df.columns and df["dG_gxtb_kcal"].notna().any():
        ng, mg, fg = _hist(df["dG_gxtb_kcal"], "Homo benzoin ΔG (g-xTB)",
                           "ΔG_g-xTB (kcal/mol)", "dG_gxtb_hist.png", "#756bb1")
        lines += ["## 3. ΔG screen (g-xTB)",
                  f"- Valid `dG_gxtb_kcal`: **{ng:,}** ({100*ng/n:.1f}%)",
                  f"- Median = **{mg:.2f}** kcal/mol; exergonic = **{fg:.1f}%**",
                  f"- Figure: `{TS}_dG_gxtb_hist.png`", ""]

        # ---- method agreement ----
        m = df[["dG_xtb_kcal", "dG_gxtb_kcal"]].dropna()
        d = m["dG_gxtb_kcal"] - m["dG_xtb_kcal"]
        r = m["dG_xtb_kcal"].corr(m["dG_gxtb_kcal"])
        plt.figure(figsize=(6.2, 6.0))
        plt.hexbin(m["dG_xtb_kcal"], m["dG_gxtb_kcal"], gridsize=80, cmap="viridis", bins="log")
        lo, hi = m.min().min(), m.max().max()
        plt.plot([lo, hi], [lo, hi], "r--", lw=1)
        plt.xlabel("ΔG_xTB (kcal/mol)"); plt.ylabel("ΔG_g-xTB (kcal/mol)")
        plt.title(f"g-xTB vs xTB  (n={len(m):,}, r={r:.3f})")
        plt.colorbar(label="log10(count)")
        plt.tight_layout(); plt.savefig(f"{ANA}/{TS}_gxtb_vs_xtb_hexbin.png", dpi=130); plt.close()
        lines += ["## 4. g-xTB vs xTB agreement",
                  f"- n paired = **{len(m):,}**, Pearson r = **{r:.3f}**",
                  f"- mean(g-xTB − xTB) = **{d.mean():.2f}** kcal/mol (bias), "
                  f"MAE = **{d.abs().mean():.2f}**, std = **{d.std():.2f}**",
                  f"- Figure: `{TS}_gxtb_vs_xtb_hexbin.png`", ""]

    rep = f"{ANA}/{TS}_REPORT.md"
    open(rep, "w").write("\n".join(lines))
    print("wrote", rep)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
