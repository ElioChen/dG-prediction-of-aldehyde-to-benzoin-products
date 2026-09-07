#!/usr/bin/env python
"""Merge Task D pilot chunks and decide the verdict: is a B97-3c Delta-learning
baseline worth a full recompute over the current g-xTB baseline?

  /home/schen3/venv/nhc-workflow/bin/python cross_benzoin/merge_cheap_baseline_pilot.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
D = REPO / "data/cross_benzoin/cheap_baseline_pilot"


def _summ(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "mean_abs": float(np.mean(np.abs(x))),
            "std": float(np.std(x)), "median_abs": float(np.median(np.abs(x))),
            "p90_abs": float(np.percentile(np.abs(x), 90)) if len(x) else None,
            "mean_signed": float(np.mean(x))}


def main() -> int:
    chunks = sorted(D.glob("chunks/cbp_*.csv"))
    if not chunks:
        print("no chunks yet")
        return 1
    df = pd.concat([pd.read_csv(f) for f in chunks], ignore_index=True).drop_duplicates("id")
    ok = df[df["error"].isna() & df["resid_gxtb"].notna() & df["resid_b973c"].notna()].copy()
    print(f"{len(ok)}/{len(df)} pilot pairs with all 3 SP levels "
          f"({df['error'].notna().sum()} errored)")

    out = {"n_total": len(df), "n_ok": len(ok),
           "conformer_noise_check": {
               "desc": "recomputed r2SCAN-3c dG on fresh ETKDG-seed42/GFN2 geom vs stored label",
               **_summ((ok["dG_r2scan_kcal"] - ok["dG_orca_kcal_stored"]).to_numpy())},
           "overall": {}, "by_group": {}}

    def _rec(sub):
        rec = {"gxtb_baseline_residual": _summ(sub["resid_gxtb"].to_numpy()),
               "b973c_baseline_residual": _summ(sub["resid_b973c"].to_numpy())}
        rg, rb = rec["gxtb_baseline_residual"], rec["b973c_baseline_residual"]
        rec["b973c_vs_gxtb"] = {
            "delta_mean_abs": rb["mean_abs"] - rg["mean_abs"],
            "delta_std": rb["std"] - rg["std"],
            "std_ratio_b973c_over_gxtb": rb["std"] / rg["std"] if rg["std"] else None}
        return rec

    out["overall"] = _rec(ok)
    for g in ("hetero_hardtail", "control"):
        sub = ok[ok["grp"] == g]
        if len(sub):
            out["by_group"][g] = _rec(sub)

    rg = out["overall"]["gxtb_baseline_residual"]
    rb = out["overall"]["b973c_baseline_residual"]
    ratio = rb["std"] / rg["std"] if rg["std"] else None
    if ratio is not None and ratio < 0.7 and rb["mean_abs"] < rg["mean_abs"] - 2.0:
        verdict = ("GREEN: B97-3c baseline residual is materially tighter "
                   f"(std x{ratio:.2f}, mean|.| {rb['mean_abs']:.2f} vs {rg['mean_abs']:.2f}). "
                   "A full B97-3c-baseline recompute + Delta-model retrain is justified.")
    elif ratio is not None and ratio < 0.9:
        verdict = ("AMBER: B97-3c helps modestly "
                   f"(std x{ratio:.2f}). Weigh full-recompute cost (~35k pairs x 3 species "
                   "B97-3c SP) against the expected MAE gain before committing.")
    else:
        verdict = ("RED: B97-3c baseline is not tighter than g-xTB "
                   f"(std x{ratio:.2f} if computed). The cheap-baseline lever is dead; "
                   "the g-xTB->r2SCAN gap is not recoverable by a cheaper-than-label SP.")
    out["verdict"] = verdict
    print("\n" + verdict)

    (D / "cheap_baseline_pilot_summary.json").write_text(json.dumps(out, indent=2, default=float))
    df.to_csv(D / "cheap_baseline_pilot_merged.csv", index=False)
    print(f"-> {D / 'cheap_baseline_pilot_summary.json'}")
    print(json.dumps(out["overall"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
