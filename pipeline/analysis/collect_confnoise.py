#!/usr/bin/env python
"""Collect the conformer-noise floor study (job 24225782, submit_confnoise.sh).

For each sampled product the worker generated K=5 conformers and ran DFT (r2SCAN-3c)
single-points, reporting the per-molecule ΔG std/range across conformers
(`dG_std_kcal`, `dG_range_kcal`). The aggregate of these per-molecule stds is the
irreducible *single-conformer label-noise floor*: no Δ-correction model trained on
single-conformer DFT labels can have a test MAE meaningfully below it, because the
target itself is uncertain by that amount.

Writes (preserve-output-history: timestamped, never overwrite):
  - confnoise_results_<ts>.csv      per-molecule concatenation
  - confnoise_summary_<ts>.md       deep markdown analysis
  - confnoise_std_hist_<ts>.png     one standalone histogram (no composite figs)
"""
import glob
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = "/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
TS = datetime.now().strftime("%Y%m%d_%H%M")

chunks = sorted(glob.glob(os.path.join(D, "confnoise_chunks", "chunk_*.csv")))
frames = []
for c in chunks:
    try:
        df = pd.read_csv(c)
        if len(df):
            frames.append(df)
    except Exception:
        pass
if not frames:
    raise SystemExit("no non-empty confnoise chunks yet")

res = pd.concat(frames, ignore_index=True).drop_duplicates("id")
res_path = os.path.join(D, f"confnoise_results_{TS}.csv")
res.to_csv(res_path, index=False)

ok = res[res["nconf_ok"] >= 2].copy()
std = ok["dG_std_kcal"].dropna()
rng = ok["dG_range_kcal"].dropna()

# An unbiased single-conformer draw has expected |error| ~= std * sqrt(2/pi) if the
# conformer ΔG is ~Gaussian around the Boltzmann mean; report both raw std and that
# proxy MAE contribution so it is directly comparable to the model test MAE (~1.6).
mae_proxy = std * np.sqrt(2.0 / np.pi)

lines = []
lines.append(f"# Conformer label-noise floor (K=5 DFT SP) — {TS}\n")
lines.append(f"- molecules attempted: **{len(res)}**, with >=2 good conformers: **{len(ok)}**")
nfail = int((res["nconf_ok"] < 2).sum())
lines.append(f"- molecules with <2 conformers (excluded): {nfail}")
if "error" in res:
    errs = res["error"].dropna()
    errs = errs[errs.astype(str).str.len() > 0]
    if len(errs):
        lines.append(f"- worker errors: {len(errs)}")
lines.append("")
lines.append("## Per-molecule ΔG spread across conformers (kcal/mol)")
lines.append("| metric | std (dG_std_kcal) | range (dG_range_kcal) |")
lines.append("|---|---|---|")
for name, q in [("mean", "mean"), ("median", "median")]:
    lines.append(f"| {name} | {getattr(std, q)():.3f} | {getattr(rng, q)():.3f} |")
lines.append(f"| 75th pct | {std.quantile(.75):.3f} | {rng.quantile(.75):.3f} |")
lines.append(f"| 90th pct | {std.quantile(.90):.3f} | {rng.quantile(.90):.3f} |")
lines.append(f"| max | {std.max():.3f} | {rng.max():.3f} |")
lines.append("")
lines.append("## Implied MAE floor from single-conformer labels")
lines.append(f"- mean per-mol ΔG std = **{std.mean():.3f}** kcal/mol")
lines.append(f"- proxy MAE contribution (std·√(2/π)) = **{mae_proxy.mean():.3f}** kcal/mol")
lines.append("")
lines.append("## Interpretation")
lines.append(
    f"The production ensemble test MAE is ~1.6 kcal/mol. The single-conformer label noise "
    f"contributes ~{mae_proxy.mean():.2f} kcal/mol (mean) of irreducible error on its own. "
    "If this is a large fraction of 1.6, the model is near the data-quality ceiling and further "
    "feature/architecture work cannot help — only better labels (Boltzmann-averaged multi-conformer "
    "DFT, or a higher-level functional) would move the floor. "
    "See [[delta-mae-noise-floor]], [[conformer-search-noise]]."
)
md_path = os.path.join(D, f"confnoise_summary_{TS}.md")
with open(md_path, "w") as f:
    f.write("\n".join(lines) + "\n")

plt.figure(figsize=(7, 5))
plt.hist(std, bins=20, color="#3b6", edgecolor="k", alpha=.8)
plt.axvline(std.mean(), color="crimson", ls="--", lw=2, label=f"mean {std.mean():.2f}")
plt.axvline(std.median(), color="navy", ls=":", lw=2, label=f"median {std.median():.2f}")
plt.xlabel("per-molecule ΔG std across 5 conformers (kcal/mol)")
plt.ylabel("count")
plt.title(f"Single-conformer DFT label-noise floor (n={len(ok)})")
plt.legend()
plt.tight_layout()
png_path = os.path.join(D, f"confnoise_std_hist_{TS}.png")
plt.savefig(png_path, dpi=140)

print("wrote:")
for p in (res_path, md_path, png_path):
    print(" ", p)
print(f"\nSUMMARY: n_ok={len(ok)} mean_std={std.mean():.3f} median_std={std.median():.3f} "
      f"mae_proxy={mae_proxy.mean():.3f} kcal/mol")
