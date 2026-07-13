#!/usr/bin/env python3
"""Controlled baseline A/B figure: GFN2 vs g-xTB semiempirical ΔG vs DFT, BOTH on the
same funnel_v3 geometry and BOTH in DMSO (g-xTB COSMO, GFN2 ALPB) — supersedes the
earlier mixed-condition semiempirical_vs_dft_dG figure. One standalone figure."""
from pathlib import Path
import glob, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path("/scratch-shared/schen3/benzoin-dg")
sys.path.insert(0, str(REPO / "pipeline")); sys.path.insert(0, str(REPO / "pipeline/compute"))
import delta_core as dc
from cho_category import cho_class
from filter_smiles import classify

df = pd.read_parquet(dc.DEFAULT_FEATURIZE_PARQUET)
g = pd.concat([pd.read_csv(f) for f in glob.glob(str(REPO/"data/raw/gxtb_baseline/chunks/chunk_*.csv"))],
              ignore_index=True)[["index", "dG_gxtb_kcal"]].dropna()
df = df.merge(g, on="index", how="inner")
df = df[~df.SMILES.map(classify).isin(dc._REACTIVE_REASONS)]
df = df[df.SMILES.map(cho_class).isin(dc.CHO_SCOPE)]
for c in ("dG_orca_kcal", "dG_xtb_kcal", "dG_gxtb_kcal"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["dG_orca_kcal", "dG_xtb_kcal", "dG_gxtb_kcal"])
df = df[df.dG_orca_kcal.abs() <= 45]

def mae(a, b): return float(np.mean(np.abs(a - b)))
def r(a, b): return float(np.corrcoef(a, b)[0, 1])
dft = df.dG_orca_kcal.to_numpy()
fig, ax = plt.subplots(figsize=(6.4, 6.2))
lim = (-45, 45); ax.plot(lim, lim, "k-", lw=1, zorder=1)
ax.scatter(df.dG_xtb_kcal, dft, s=10, alpha=0.35, color="#888888", zorder=2,
           label=f"GFN2  (MAE {mae(df.dG_xtb_kcal.to_numpy(),dft):.1f}, r {r(df.dG_xtb_kcal.to_numpy(),dft):.2f})")
ax.scatter(df.dG_gxtb_kcal, dft, s=10, alpha=0.55, color="#1f6fb2", zorder=3,
           label=f"g-xTB (MAE {mae(df.dG_gxtb_kcal.to_numpy(),dft):.1f}, r {r(df.dG_gxtb_kcal.to_numpy(),dft):.2f})")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("semiempirical reaction ΔG  [kcal/mol]")
ax.set_ylabel("DFT r2SCAN-3c reaction ΔG  [kcal/mol]")
ax.set_title("Semiempirical vs DFT reaction ΔG — controlled\n"
             f"(same funnel_v3 geometry, both DMSO; n={len(df)} in-scope)", fontsize=11)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
fig.tight_layout()
out = REPO/"figs/baseline_gxtb_vs_gfn2_controlled_20260622.png"
fig.savefig(out, dpi=150); print("wrote", out)
