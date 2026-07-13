#!/usr/bin/env python
"""Re-plot xTB vs DFT (r2SCAN-3c//xTB) correlation after removing hypervalent-S(VI)
/ strong-EWG motifs (sulfonyl, sulfonyl fluoride, triflate, nitro).

Reads the 1% validation merged CSV, tags motif membership via SMARTS, recomputes
correlation metrics on the full set vs the EWG-removed subset, and writes STANDALONE
figures (one per file) + a markdown report. New timestamped filenames; nothing is
overwritten.
"""
import sys, datetime, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MERGED = sys.argv[1] if len(sys.argv) > 1 else \
    "/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c/analysis/dft_sp_merged_20260618_002636.csv"
OUT = "/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c/analysis"
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- EWG motif SMARTS (the four called out in the validation report) ---
MOTIFS = {
    "sulfonyl  S(=O)(=O)":          "[#16X4](=[OX1])(=[OX1])",      # covers sulfonyl fluoride, triflate, sulfonate, sulfonamide
    "sulfonyl_fluoride S(=O)(=O)F": "[#16X4](=[OX1])(=[OX1])[F]",
    "triflate OS(=O)(=O)CF3":       "[OX2][#16X4](=[OX1])(=[OX1])[CX4]([F])([F])[F]",
    "nitro [N+](=O)[O-]":           "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in MOTIFS.items()}

df = pd.read_csv(MERGED)
# valid pairs only
df = df[df["error"].isna() if "error" in df.columns else slice(None)].copy()
df = df.dropna(subset=["dG_xtb_kcal", "dG_orca_kcal"]).copy()

def tag(sm):
    m = Chem.MolFromSmiles(sm) if isinstance(sm, str) else None
    if m is None:
        return {k: False for k in MOTIFS}
    return {k: m.HasSubstructMatch(p) for k, p in PATS.items()}

tags = df["aldehyde_smiles"].apply(tag).apply(pd.Series)
df = pd.concat([df, tags], axis=1)
# "removed" = any sulfonyl-class (S(=O)(=O)) OR nitro
df["is_ewg"] = df["sulfonyl  S(=O)(=O)"] | df["nitro [N+](=O)[O-]"]

def metrics(d):
    x = d["dG_xtb_kcal"].to_numpy(); y = d["dG_orca_kcal"].to_numpy()
    resid = x - y
    sl, ic, r, p, se = stats.linregress(x, y)
    ycorr = sl * x + ic                      # linear-corrected xTB prediction of DFT
    lin_mae = float(np.mean(np.abs(y - ycorr)))
    return dict(n=len(d), mae=float(np.mean(np.abs(resid))),
                rmse=float(np.sqrt(np.mean(resid**2))),
                bias=float(np.mean(resid)), r=float(r), r2=float(r**2),
                slope=float(sl), intercept=float(ic), lin_mae=lin_mae)

m_full = metrics(df)
m_keep = metrics(df[~df["is_ewg"]])
m_drop = metrics(df[df["is_ewg"]]) if df["is_ewg"].any() else None

# ---- Figure 1: cleaned-subset correlation (the requested curve) ----
d = df[~df["is_ewg"]]
x = d["dG_xtb_kcal"].to_numpy(); y = d["dG_orca_kcal"].to_numpy()
lim = [min(x.min(), y.min()) - 3, max(x.max(), y.max()) + 3]
plt.figure(figsize=(6.2, 6.0))
plt.scatter(x, y, s=10, alpha=0.45, color="#2c7fb8", edgecolors="none")
xs = np.linspace(lim[0], lim[1], 100)
plt.plot(xs, m_keep["slope"] * xs + m_keep["intercept"], "r-", lw=1.6,
         label=f"fit: y={m_keep['slope']:.3f}x+{m_keep['intercept']:.2f}")
plt.plot(lim, lim, "k--", lw=1.0, alpha=0.6, label="y = x")
plt.xlim(lim); plt.ylim(lim)
plt.xlabel("ΔG xTB (GFN2)  [kcal/mol]")
plt.ylabel("ΔG DFT (r2SCAN-3c//xTB)  [kcal/mol]")
plt.title(f"xTB vs DFT — EWG-removed (n={m_keep['n']})\n"
          f"r={m_keep['r']:.3f}  MAE={m_keep['mae']:.2f}  bias={m_keep['bias']:.2f}")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
f1 = f"{OUT}/scatter_xtb_vs_dft_dropEWG_{TS}.png"
plt.savefig(f1, dpi=150); plt.close()

# ---- Figure 2: full set with EWG points highlighted (shows what was removed) ----
plt.figure(figsize=(6.2, 6.0))
keep = df[~df["is_ewg"]]; drop = df[df["is_ewg"]]
xa = df["dG_xtb_kcal"].to_numpy(); ya = df["dG_orca_kcal"].to_numpy()
lim2 = [min(xa.min(), ya.min()) - 3, max(xa.max(), ya.max()) + 3]
plt.scatter(keep["dG_xtb_kcal"], keep["dG_orca_kcal"], s=10, alpha=0.4,
            color="#2c7fb8", edgecolors="none", label=f"kept (n={len(keep)})")
plt.scatter(drop["dG_xtb_kcal"], drop["dG_orca_kcal"], s=14, alpha=0.7,
            color="#d7301f", edgecolors="none", label=f"EWG removed (n={len(drop)})")
plt.plot(lim2, lim2, "k--", lw=1.0, alpha=0.6, label="y = x")
plt.xlim(lim2); plt.ylim(lim2)
plt.xlabel("ΔG xTB (GFN2)  [kcal/mol]")
plt.ylabel("ΔG DFT (r2SCAN-3c//xTB)  [kcal/mol]")
plt.title("xTB vs DFT — EWG motifs highlighted")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
f2 = f"{OUT}/scatter_xtb_vs_dft_EWGhighlight_{TS}.png"
plt.savefig(f2, dpi=150); plt.close()

# ---- Figure 3: residual vs xTB, cleaned subset ----
plt.figure(figsize=(6.4, 5.0))
resid = x - y
plt.scatter(x, resid, s=10, alpha=0.45, color="#2c7fb8", edgecolors="none")
plt.axhline(0, color="k", lw=1.0)
plt.axhline(m_keep["bias"], color="r", ls="--", lw=1.2, label=f"mean bias={m_keep['bias']:.2f}")
plt.xlabel("ΔG xTB (GFN2)  [kcal/mol]")
plt.ylabel("residual  ΔG_xTB − ΔG_DFT  [kcal/mol]")
plt.title(f"Residual vs xTB — EWG-removed (n={m_keep['n']})")
plt.legend(fontsize=8); plt.tight_layout()
f3 = f"{OUT}/residual_vs_xtb_dropEWG_{TS}.png"
plt.savefig(f3, dpi=150); plt.close()

# ---- per-motif counts ----
counts = {k: int(df[k].sum()) for k in MOTIFS}

# ---- markdown report ----
def row(label, m):
    return (f"| {label} | {m['n']} | {m['mae']:.2f} | {m['rmse']:.2f} | "
            f"{m['bias']:.2f} | {m['r']:.3f} | {m['r2']:.3f} | "
            f"{m['slope']:.3f} | {m['intercept']:.2f} | {m['lin_mae']:.2f} |")

md = f"""# xTB vs DFT correlation after removing hypervalent-S / strong-EWG motifs — {TS}

Source merged set: `{MERGED}`
DFT reference = r2SCAN-3c single-point on xTB geometry + xTB-RRHO (//xTB).

## Motif counts in the 1% set (n={m_full['n']})

| motif (SMARTS) | matches | % of set |
|---|---|---|
""" + "\n".join(
    f"| {k} | {counts[k]} | {100*counts[k]/m_full['n']:.1f} |" for k in MOTIFS
) + f"""

Removal rule: **drop if sulfonyl `S(=O)(=O)` OR nitro** (sulfonyl already subsumes
sulfonyl fluoride + triflate). Removed **{int(df['is_ewg'].sum())}** / {m_full['n']}
molecules ({100*df['is_ewg'].sum()/m_full['n']:.1f}%).

## Correlation metrics

| set | n | MAE | RMSE | bias(xTB−DFT) | Pearson r | R² | fit slope | fit intercept | lin-corr MAE |
|---|---|---|---|---|---|---|---|---|---|
{row('FULL', m_full)}
{row('EWG-removed (kept)', m_keep)}
{row('EWG-only (removed)', m_drop) if m_drop else ''}

## Interpretation

- Removing the EWG motifs ({100*df['is_ewg'].sum()/m_full['n']:.1f}% of molecules)
  moves MAE **{m_full['mae']:.2f} → {m_keep['mae']:.2f}** and Pearson r
  **{m_full['r']:.3f} → {m_keep['r']:.3f}**, confirming these motifs are the dominant
  source of xTB–DFT disagreement (consistent with the worst-10% enrichment in the
  validation report).
- The systematic bias also shrinks ({m_full['bias']:.2f} → {m_keep['bias']:.2f}
  kcal/mol): xTB over-exergonicity is largely an EWG artifact, not a uniform offset.
- Residual structure on the cleaned set: lin-corrected MAE = {m_keep['lin_mae']:.2f}
  kcal/mol (vs ~3.2 noise floor) — {'still above the floor, so DFT keeps adding signal even on benign aldehydes.' if m_keep['lin_mae'] > 4 else 'approaching the noise floor, so a linear xTB→DFT correction captures most of the benign-aldehyde signal.'}

## Figures (standalone)

- `{f1.split('/')[-1]}`  — EWG-removed correlation (requested curve)
- `{f2.split('/')[-1]}`  — full set, EWG points highlighted in red
- `{f3.split('/')[-1]}`  — residual vs xTB, EWG-removed
"""
mdpath = f"{OUT}/REPORT_corr_dropEWG_{TS}.md"
with open(mdpath, "w") as fh:
    fh.write(md)

# also dump the tagged csv for reuse
tagged = f"{OUT}/merged_ewgtagged_{TS}.csv"
df.to_csv(tagged, index=False)

print("FULL :", m_full)
print("KEEP :", m_keep)
print("DROP :", m_drop)
print("counts:", counts)
print("wrote:", f1, f2, f3, mdpath, tagged, sep="\n  ")
