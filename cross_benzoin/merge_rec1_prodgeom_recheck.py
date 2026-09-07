#!/usr/bin/env python
"""Merge + analyse the Rec-1 production-geometry recheck chunks.

Core question: on the REAL production funnel_v3 + GFN2-ohess geometry protocol, does the
D-pilot's low B97-3c residual scatter survive?
  pilot (one-shot ETKDG-seed42 geom, 128 pairs): resid_gxtb std 4.32, resid_b973c std 1.11 (ratio 0.26)
  pilot, the 30 picked pairs only                : resid_gxtb std 4.51, resid_b973c std 1.42

Prints std(resid_gxtb) vs std(resid_b973c) on the production geometry, plus repro checks
(recomputed r2SCAN / g-xTB dG vs the stored production labels -> funnel_v3 reproduction
fidelity / conformer noise).
"""
from __future__ import annotations
import glob
import json
import statistics as st
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
D = REPO / "data/cross_benzoin/rec1_prodgeom_recheck"
PILOT30_GXTB_STD, PILOT30_B973C_STD = 4.51, 1.42
PILOT_ALL_GXTB_STD, PILOT_ALL_B973C_STD = 4.32, 1.11


def _s(xs):
    xs = [float(x) for x in xs if pd.notna(x)]
    if not xs:
        return dict(n=0)
    a = sorted(abs(x) for x in xs)
    return dict(n=len(xs), mean=round(st.mean(xs), 3), mean_abs=round(st.mean(a), 3),
                std=round(st.pstdev(xs), 3), median_abs=round(a[len(a) // 2], 3),
                p90_abs=round(a[max(0, int(0.9 * len(a)) - 1)], 3))


def main() -> int:
    files = sorted(glob.glob(str(D / "chunks/rec1_*.csv")))
    if not files:
        print("no chunk csvs yet")
        return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(D / "rec1_prodgeom_recheck_merged.csv", index=False)
    ok = df[df["error"].isna() & df["resid_b973c"].notna()].copy()
    print(f"{len(files)} chunks -> {len(df)} rows, {len(ok)} usable, {len(df) - len(ok)} errored")
    if len(df) - len(ok):
        for _, r in df[df["error"].notna()].iterrows():
            print(f"  ERR {r['pid']}: {r['error']}")

    out = {"n_chunks": len(files), "n_rows": len(df), "n_ok": len(ok),
           "pilot_ref": {"all128": {"resid_gxtb_std": PILOT_ALL_GXTB_STD, "resid_b973c_std": PILOT_ALL_B973C_STD},
                         "picked30": {"resid_gxtb_std": PILOT30_GXTB_STD, "resid_b973c_std": PILOT30_B973C_STD}},
           "prod_geom": {}}
    for name, sub in [("overall", ok)] + [(g, ok[ok["grp"] == g]) for g in sorted(ok["grp"].unique())]:
        blk = {k: _s(sub[k]) for k in ["resid_gxtb", "resid_b973c", "repro_r2scan", "repro_gxtb"]}
        rg, rb = blk["resid_gxtb"].get("std"), blk["resid_b973c"].get("std")
        blk["std_ratio_b973c_over_gxtb"] = round(rb / rg, 3) if rg else None
        out["prod_geom"][name] = blk
        print(f"\n=== {name} (n={len(sub)}) ===")
        for k in ["resid_gxtb", "resid_b973c", "repro_r2scan", "repro_gxtb"]:
            print(f"  {k:13s}: {blk[k]}")
        print(f"  std ratio b973c/gxtb (prod geom): {blk['std_ratio_b973c_over_gxtb']}  "
              f"(pilot all128 0.26 / picked30 {PILOT30_B973C_STD / PILOT30_GXTB_STD:.2f})")

    ov = out["prod_geom"]["overall"]
    rb = ov["resid_b973c"].get("std")
    rg = ov["resid_gxtb"].get("std")
    repro = ov["repro_r2scan"].get("std")
    ratio = ov["std_ratio_b973c_over_gxtb"]
    notes = []
    if repro is not None and repro > 3.5:
        notes.append(f"repro_r2scan std {repro} > 3.5: funnel_v3 regen not reproducing the "
                     f"stored labels well -> residuals partly contaminated by geometry-regen noise; "
                     f"interpret ratio cautiously.")
    if ratio is not None:
        if ratio <= 0.45 and rb is not None and rb <= 2.5:
            verdict = (f"GREEN: B97-3c keeps low scatter on the production geometry "
                       f"(std {rb} vs g-xTB {rg}, ratio {ratio}). The pilot's 0.26 was not a "
                       f"one-shot-conformer artefact. -> proceed to full B97-3c-baseline recompute + Delta-model retrain.")
        elif ratio <= 0.7:
            verdict = (f"AMBER: B97-3c still helps on the production geometry (ratio {ratio}, "
                       f"std {rb} vs {rg}) but less decisively than the pilot's 0.26. "
                       f"Weigh full-recompute cost against the smaller expected floor drop.")
        else:
            verdict = (f"RED: B97-3c advantage largely gone on the production geometry "
                       f"(ratio {ratio}, std {rb} vs {rg}). The pilot's 1.11 was a conformer "
                       f"artefact of the shared one-shot geometry. Lever weak -> do not commit to full recompute.")
    else:
        verdict = "INCONCLUSIVE: could not compute std ratio."
    out["verdict"] = verdict
    out["notes"] = notes
    print(f"\n>>> {verdict}")
    for n in notes:
        print(f"  note: {n}")
    (D / "rec1_prodgeom_recheck_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {D / 'rec1_prodgeom_recheck_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
