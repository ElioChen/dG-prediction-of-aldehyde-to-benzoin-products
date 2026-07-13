#!/usr/bin/env python
"""Collect the better-label probe (job 24226784, submit_boltz_relabel.sh).

Answers: would multi-conformer Boltzmann-averaged DFT, or a higher-level functional
(wB97X-3c), move the ~1.6 kcal/mol MAE floor? On 120 held-out *test* molecules we
recomputed the electronic reaction energy three ways and form corrected labels by
applying each ΔE shift to the stored label (thermal term unchanged):

  dG_boltz = dG_stored + (dE_r2_boltz  - dE_r2_single)   # multi-conformer Boltzmann
  dG_wb97x = dG_stored + (dE_wb_single - dE_r2_single)   # functional swap

Then we re-score the *unchanged* production model predictions against each label set.
If MAE(pred, dG_boltz) < MAE(pred, dG_stored), the single-conformer label noise was
inflating the measured MAE and better labels would lower the floor; by how much
quantifies the achievable gain.

Writes (timestamped; never overwrite):
  boltz_relabel_results_<ts>.csv, boltz_relabel_summary_<ts>.md,
  boltz_relabel_corr_hist_<ts>.png, boltz_relabel_func_hist_<ts>.png
"""
import glob, os
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

D = "/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
H = "/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6"
TS = datetime.now().strftime("%Y%m%d_%H%M")

frames = []
for c in sorted(glob.glob(os.path.join(D, "boltz_chunks", "chunk_*.csv"))):
    try:
        df = pd.read_csv(c)
        if len(df):
            frames.append(df)
    except Exception:
        pass
if not frames:
    raise SystemExit("no non-empty boltz chunks yet")
res = pd.concat(frames, ignore_index=True).drop_duplicates("id")

samp = pd.read_csv(os.path.join(D, "boltz_relabel_sample.csv"))[["id", "dG_orca_kcal", "dG_gxtb_kcal"]]
pred = pd.read_csv(os.path.join(D, "products_dG_corrected_FINAL_20260626.csv"))[["id", "dG_gxtb_corrected_final"]]
m = res.merge(samp, on="id").merge(pred, on="id")
ok = m[m["dE_r2_single"].notna() & m["dE_r2_boltz"].notna()].copy()

ok["boltz_corr"] = ok["dE_r2_boltz"] - ok["dE_r2_single"]
ok["func_shift"] = ok["dE_wb_single"] - ok["dE_r2_single"]
ok["dG_boltz"] = ok["dG_orca_kcal"] + ok["boltz_corr"]
ok["dG_wb97x"] = ok["dG_orca_kcal"] + ok["func_shift"]
P = ok["dG_gxtb_corrected_final"]
mae = lambda y: float((P - y).abs().mean())
mae_stored = mae(ok["dG_orca_kcal"])
mae_boltz = mae(ok["dG_boltz"])
wb = ok[ok["func_shift"].notna()]
mae_wb = float((wb["dG_gxtb_corrected_final"] - wb["dG_wb97x"]).abs().mean()) if len(wb) else float("nan")

ok.to_csv(os.path.join(D, f"boltz_relabel_results_{TS}.csv"), index=False)

bc, fs = ok["boltz_corr"], ok["func_shift"].dropna()
L = []
L.append(f"# Better-label probe — Boltzmann DFT & wB97X-3c — {TS}\n")
L.append(f"n molecules (both species ok): **{len(ok)}** of 120 test-split; wB97X-3c ok: {len(fs)}\n")
L.append("## How much do labels move? (kcal/mol)")
L.append("| shift | mean | std | mean|·| | 90th pct |·| | max |·| |")
L.append("|---|---|---|---|---|---|")
L.append(f"| Boltzmann correction (boltz−single) | {bc.mean():.3f} | {bc.std():.3f} | "
         f"{bc.abs().mean():.3f} | {bc.abs().quantile(.9):.3f} | {bc.abs().max():.3f} |")
if len(fs):
    L.append(f"| functional shift (wB97X−r2SCAN) | {fs.mean():.3f} | {fs.std():.3f} | "
             f"{fs.abs().mean():.3f} | {fs.abs().quantile(.9):.3f} | {fs.abs().max():.3f} |")
L.append("")
L.append("## Does re-labeling lower the model MAE on these 120? (same frozen predictions)")
L.append("| label set | model MAE |")
L.append("|---|---|")
L.append(f"| stored single-conformer r2SCAN-3c (current) | **{mae_stored:.3f}** |")
L.append(f"| Boltzmann-averaged r2SCAN-3c | **{mae_boltz:.3f}** ({mae_boltz - mae_stored:+.3f}) |")
if not np.isnan(mae_wb):
    L.append(f"| wB97X-3c single-conformer | **{mae_wb:.3f}** ({mae_wb - mae_stored:+.3f}) |")
L.append("")
L.append("## Interpretation")
verdict = ("LOWERS" if mae_boltz < mae_stored - 0.05 else
           "does NOT lower" if mae_boltz > mae_stored - 0.05 else "marginally changes")
L.append(
    f"Multi-conformer Boltzmann re-labeling **{verdict}** the measured MAE "
    f"({mae_stored:.2f}→{mae_boltz:.2f}). The Boltzmann correction has std {bc.std():.2f} and "
    f"mean magnitude {bc.abs().mean():.2f} kcal/mol — that is the label movement available from "
    "conformer averaging. The functional swap (wB97X-3c vs r2SCAN-3c) moves labels by "
    f"std {fs.std():.2f} (mean |·| {fs.abs().mean():.2f}) — a systematic+random bias of the chosen "
    "functional. If either MAE drops materially below 1.6, single-conformer/functional label noise "
    "is a real component of the floor and a full multi-conformer (or higher-functional) re-label is "
    "justified; if not, the floor is intrinsic model/feature error. "
    "See [[delta-mae-noise-floor]], [[conformer-search-noise]], [[dft-labels-r2scan-not-pbe0]]."
)
with open(os.path.join(D, f"boltz_relabel_summary_{TS}.md"), "w") as f:
    f.write("\n".join(L) + "\n")

for col, fn, lab in [("boltz_corr", "corr", "Boltzmann correction  ΔE_boltz − ΔE_single"),
                     ("func_shift", "func", "functional shift  wB97X-3c − r2SCAN-3c")]:
    v = ok[col].dropna()
    if not len(v):
        continue
    plt.figure(figsize=(7, 5))
    plt.hist(v, bins=20, color="#46c", edgecolor="k", alpha=.8)
    plt.axvline(0, color="k", lw=1)
    plt.axvline(v.mean(), color="crimson", ls="--", lw=2, label=f"mean {v.mean():.2f}")
    plt.xlabel(f"{lab} (kcal/mol)"); plt.ylabel("count")
    plt.title(f"{lab}\nstd={v.std():.2f}  n={len(v)}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(D, f"boltz_relabel_{fn}_hist_{TS}.png"), dpi=140)

print(f"n_ok={len(ok)}  MAE stored={mae_stored:.3f} boltz={mae_boltz:.3f} wb97x={mae_wb:.3f}")
print(f"boltz_corr std={bc.std():.3f} mean|.|={bc.abs().mean():.3f} | func_shift std={fs.std():.3f}")
print("wrote summary/results/figs with ts", TS)
