#!/usr/bin/env python
"""Standalone visualization for deep_error_analysis_champion275.py's noise-band accounting
(section 1 of REPORT_deep_error_analysis_champion275_20260707.md): a histogram of per-molecule
test error split into 'noise-explainable' (<= 3-sigma of the established seed-reshuffle noise
band) vs 'genuinely elevated' (model failing to capture, not just label jitter). The original
script computed this ratio (67.7% / 32.3%) but never plotted the distribution itself -- this
closes that gap. One standalone figure, no composite panels.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

H = "/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6"
OUT = f"{H}/viz_gxtb_20260625"
NOISE_MEAN, NOISE_STD = 1.571, 0.013  # baseline_72 robustness study, REPORT_robustness_baseline72_20260702.md
noise_hi = NOISE_MEAN + 3 * NOISE_STD

df = pd.read_csv(f"{OUT}/test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv", usecols=["id", "error"])
err = df["error"].values
frac_within = float((err <= noise_hi).mean())
n = len(err)

fig, ax = plt.subplots(figsize=(8, 5.5))
bins = np.linspace(0, np.percentile(err, 99.5), 80)
within = err[err <= noise_hi]; above = err[err > noise_hi]
ax.hist(within, bins=bins, color="#4C72B0", alpha=0.85, label=f"within noise band ({frac_within*100:.1f}%, n={len(within):,})")
ax.hist(above, bins=bins, color="#C44E52", alpha=0.85, label=f"genuinely elevated ({(1-frac_within)*100:.1f}%, n={len(above):,})")
ax.axvline(NOISE_MEAN, color="k", ls="-", lw=1.2, label=f"noise-band mean ({NOISE_MEAN:.3f})")
ax.axvline(noise_hi, color="k", ls="--", lw=1.4, label=f"3-sigma cutoff ({noise_hi:.3f})")
ax.set_xlabel("test-set absolute error (kcal/mol)")
ax.set_ylabel("count")
ax.set_title(f"MORDREDSLIM271_BDEGXTB: real error vs noise band (n={n:,} test molecules)")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/126_noise_band_histogram_champion275_20260707.png", dpi=150, bbox_inches="tight")
print(f"saved 126_noise_band_histogram_champion275_20260707.png | within={frac_within*100:.1f}% above={  (1-frac_within)*100:.1f}%")
