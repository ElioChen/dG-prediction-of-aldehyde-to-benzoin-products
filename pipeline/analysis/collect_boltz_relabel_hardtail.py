#!/usr/bin/env python
"""Collect the hard-tail Boltzmann relabel probe (2026-07-10, job 24528107,
submit_boltz_relabel_hardtail.sh). Corrected version of collect_boltz_relabel.py: targets the
actual sulfonyl/P/imine/amide error-driving subset (not a random draw), uses K=10 conformers
(not 5), and scores against the CURRENT champion's own dG_pred (not a stale June model).

Run only after the boltz_chunks_hardtail_20260710/ array has finished (check with
`squeue -u $USER` / count of non-empty chunk_*.csv files vs 150 expected rows).
"""
import glob, os
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

D = "/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
TS = datetime.now().strftime("%Y%m%d_%H%M")

frames = []
for c in sorted(glob.glob(os.path.join(D, "boltz_chunks_hardtail_20260710", "chunk_*.csv"))):
    try:
        df = pd.read_csv(c)
        if len(df):
            frames.append(df)
    except Exception:
        pass
if not frames:
    raise SystemExit("no non-empty hard-tail boltz chunks yet")
res = pd.concat(frames, ignore_index=True).drop_duplicates("id")
print(f"collected {len(res)} rows from {len(frames)} chunks", flush=True)

samp = pd.read_csv(os.path.join(D, "boltz_relabel_hardtail_sample_20260710.csv"))
# samp already has dG_pred = CURRENT champion's own prediction for these ids (not a stale model)
m = res.merge(samp, on="id")
ok = m[m["dE_r2_single"].notna() & m["dE_r2_boltz"].notna()].copy()
print(f"n_ok (both species converged)={len(ok)} / sample n={len(samp)}", flush=True)

ok["boltz_corr"] = ok["dE_r2_boltz"] - ok["dE_r2_single"]
ok["func_shift"] = ok["dE_wb_single"] - ok["dE_r2_single"]
ok["dG_boltz"] = ok["dG_orca_kcal"] + ok["boltz_corr"]
ok["dG_wb97x"] = ok["dG_orca_kcal"] + ok["func_shift"]
P = ok["dG_pred"]
mae = lambda y: float((P - y).abs().mean())
mae_stored = mae(ok["dG_orca_kcal"])
mae_boltz = mae(ok["dG_boltz"])
wb = ok[ok["func_shift"].notna()]
mae_wb = float((wb["dG_pred"] - wb["dG_wb97x"]).abs().mean()) if len(wb) else float("nan")

ok.to_csv(os.path.join(D, f"boltz_relabel_hardtail_results_{TS}.csv"), index=False)

bc, fs = ok["boltz_corr"], ok["func_shift"].dropna()
L = []
L.append(f"# Better-label probe, HARD-TAIL TARGETED (K=10 conformers) — {TS}\n")
L.append("Corrected version of the 2026-06-26 pilot (see descriptor-search-exhausted memory's "
         "2026-07-10 correction): targets the sulfonyl/P/imine/amide top-|error| subset of the "
         "CURRENT champion's test set (not a random draw of easy molecules), K=10 conformers "
         "(not 5), scored against the current champion's own dG_pred (not a stale June model).\n")
L.append(f"\nn molecules (both species ok): **{len(ok)}** of {len(samp)} targeted hard-tail sample; "
        f"wB97X-3c ok: {len(fs)}\n")
L.append("## How much do labels move? (kcal/mol)")
L.append("| shift | mean | std | mean|·| | 90th pct |·| | max |·| |")
L.append("|---|---|---|---|---|---|")
L.append(f"| Boltzmann correction (boltz−single) | {bc.mean():.3f} | {bc.std():.3f} | "
         f"{bc.abs().mean():.3f} | {bc.abs().quantile(.9):.3f} | {bc.abs().max():.3f} |")
if len(fs):
    L.append(f"| functional shift (wB97X−r2SCAN) | {fs.mean():.3f} | {fs.std():.3f} | "
             f"{fs.abs().mean():.3f} | {fs.abs().quantile(.9):.3f} | {fs.abs().max():.3f} |")
L.append("")
L.append("## Does re-labeling lower the CURRENT champion's MAE on this hard-tail sample? (frozen predictions)")
L.append("| label set | model MAE |")
L.append("|---|---|")
L.append(f"| stored single-conformer r2SCAN-3c (current) | **{mae_stored:.3f}** |")
L.append(f"| Boltzmann-averaged r2SCAN-3c (K=10) | **{mae_boltz:.3f}** ({mae_boltz - mae_stored:+.3f}) |")
if not np.isnan(mae_wb):
    L.append(f"| wB97X-3c single-conformer | **{mae_wb:.3f}** ({mae_wb - mae_stored:+.3f}) |")
L.append("")
L.append("## Interpretation")
verdict = ("LOWERS" if mae_boltz < mae_stored - 0.05 else
           "does NOT lower" if mae_boltz > mae_stored - 0.05 else "marginally changes")
L.append(
    f"Multi-conformer Boltzmann re-labeling **{verdict}** the champion's measured MAE on this "
    f"hard-tail sample ({mae_stored:.2f}→{mae_boltz:.2f}). Caveat unchanged from the original "
    "pilot: this re-scores FROZEN predictions rather than retraining, so it only tests whether "
    "label noise is inflating apparent error on molecules the model already got wrong -- it does "
    "NOT directly measure whether a model retrained on relabeled data would generalize better. "
    "If this LOWERS MAE meaningfully (materially more than the -0.05 threshold), that is real "
    "evidence the label-noise lever is worth pursuing at scale (full-tail relabel + retrain); if "
    "it does not, the two independent nulls (random-draw K=5 and targeted-hard-tail K=10) would "
    "jointly rule out this lever and it should be dropped for good."
)
with open(os.path.join(D, f"boltz_relabel_hardtail_summary_{TS}.md"), "w") as f:
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
    plt.title(f"hard-tail {lab}\nstd={v.std():.2f}  n={len(v)}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(D, f"boltz_relabel_hardtail_{fn}_hist_{TS}.png"), dpi=140)

print(f"n_ok={len(ok)}  MAE stored={mae_stored:.3f} boltz={mae_boltz:.3f} wb97x={mae_wb:.3f}")
print(f"boltz_corr std={bc.std():.3f} mean|.|={bc.abs().mean():.3f} | func_shift std={fs.std():.3f}")
print("wrote summary/results/figs with ts", TS)
