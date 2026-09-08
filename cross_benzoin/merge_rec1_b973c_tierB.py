#!/usr/bin/env python
"""Merge + QC the Tier B self-consistent B97-3c relabel shards.

Tier B worker writes one row per pair to shards/shard_%05d.csv (append + fsync,
crash-safe). This merges them into one table keyed on `pid`, reports coverage vs the
35,528-pair master list, and prints the residual-scatter QC that decides whether the
B97-3c baseline lever survived at full scale:

  A/B proxy refs (HANDOFF sec 5): B97-3c Delta-model holdout MAE 0.652 (self-consistent) /
  1.694 (stock) vs g-xTB 1.88 / 2.665; resid std ratio target <= 0.45 (GREEN).

Outputs:
  <prefix>_merged.csv    deduped full table (one row per pid, real result preferred)
  <prefix>_labels.csv    slim pid,dG_r2scan_kcal,dG_b973c_kcal,dG_gxtb_kcal for the assembler
  <prefix>_summary.json   coverage + per-group residual/repro stats + verdict

Run any time during the campaign for a progress QC; run at drain for the real merge
(HANDOFF sec 1.3 step 1).
"""
from __future__ import annotations
import glob
import json
import statistics as st
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
D = REPO / "data/cross_benzoin/rec1_b973c_tierB"
MASTER = D / "rec1_b973c_tierB_pairs_35528.csv"
PILOT_GXTB_STD, PILOT_B973C_STD = 4.32, 1.11   # D-pilot one-shot-geom refs


def _s(xs):
    xs = [float(x) for x in xs if pd.notna(x)]
    if not xs:
        return dict(n=0)
    a = sorted(abs(x) for x in xs)
    return dict(n=len(xs), mean=round(st.mean(xs), 3), mean_abs=round(st.mean(a), 3),
                std=round(st.pstdev(xs), 3), median_abs=round(a[len(a) // 2], 3),
                p90_abs=round(a[max(0, int(0.9 * len(a)) - 1)], 3))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-glob", default=str(D / "shards/shard_*.csv"))
    ap.add_argument("--out-prefix", default=str(D / "rec1_b973c_tierB"))
    ap.add_argument("--master", default=str(MASTER))
    args = ap.parse_args()

    files = sorted(glob.glob(args.shards_glob))
    if not files:
        print("no shard csvs yet")
        return 1
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, dtype={"pid": str}))
        except Exception as e:  # a shard mid-append can have a torn last line
            parts.append(pd.read_csv(f, dtype={"pid": str}, on_bad_lines="skip"))
            print(f"  warn: {Path(f).name} parsed with on_bad_lines=skip ({e})")
    df = pd.concat(parts, ignore_index=True)
    df["pid"] = df["pid"].astype(str).str.strip()
    for c in ("dG_r2scan_kcal", "dG_b973c_kcal", "dG_gxtb_kcal",
              "resid_gxtb", "resid_b973c", "repro_r2scan", "repro_gxtb"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # dedup: one row per pid, prefer a real result (dG_r2scan_kcal present)
    df["_has"] = df["dG_r2scan_kcal"].notna().astype(int)
    df = (df.sort_values(["pid", "_has"]).drop_duplicates("pid", keep="last")
            .drop(columns="_has").reset_index(drop=True))

    err = df["error"].astype(str).str.strip()
    is_err = err.ne("") & err.ne("nan")
    ok = df[df["dG_r2scan_kcal"].notna() & ~is_err].copy()
    true_fail = df[df["dG_r2scan_kcal"].isna()]
    df.to_csv(f"{args.out_prefix}_merged.csv", index=False)
    ok[["pid", "dG_r2scan_kcal", "dG_b973c_kcal", "dG_gxtb_kcal"]].to_csv(
        f"{args.out_prefix}_labels.csv", index=False)

    master = pd.read_csv(args.master, dtype=str)
    mcol = "pid" if "pid" in master.columns else master.columns[0]
    mids = set(master[mcol].astype(str).str.strip())
    got = set(ok["pid"])
    cov = len(got & mids)

    print(f"{len(files)} shards -> {len(df)} unique pid, {len(ok)} usable, "
          f"{len(true_fail)} true-fail (dG_r2scan empty), {int(is_err.sum())} error-tagged")
    print(f"coverage vs master ({len(mids)}): {cov} ({100*cov/len(mids):.1f}%), "
          f"{len(mids)-cov} remaining")
    for _, r in true_fail.head(15).iterrows():
        print(f"  FAIL {r['pid']}: error={r.get('error')!r}")

    out = {"n_shards": len(files), "n_unique_pid": len(df), "n_ok": len(ok),
           "n_true_fail": int(len(true_fail)), "n_error_tagged": int(is_err.sum()),
           "master_n": len(mids), "coverage": cov, "coverage_pct": round(100*cov/len(mids), 2),
           "pilot_ref_same_pairs": {"resid_gxtb_std": PILOT_GXTB_STD,
                                    "resid_b973c_std": PILOT_B973C_STD,
                                    "std_ratio": round(PILOT_B973C_STD / PILOT_GXTB_STD, 3)},
           "groups": {}}
    grp_col = "new_scaffold_split" if "new_scaffold_split" in ok.columns else None
    blocks = [("overall", ok)]
    if grp_col:
        blocks += [(g, ok[ok[grp_col] == g]) for g in sorted(ok[grp_col].dropna().unique())]
    for name, sub in blocks:
        blk = {k: _s(sub[k]) for k in ("resid_gxtb", "resid_b973c", "repro_r2scan", "repro_gxtb")
               if k in sub.columns}
        rg = blk.get("resid_gxtb", {}).get("std")
        rb = blk.get("resid_b973c", {}).get("std")
        blk["std_ratio_b973c_over_gxtb"] = round(rb / rg, 3) if rg else None
        out["groups"][name] = blk
        print(f"\n=== {name} (n={len(sub)}) ===")
        for k in ("resid_gxtb", "resid_b973c", "repro_r2scan", "repro_gxtb"):
            if k in blk:
                print(f"  {k:13s}: {blk[k]}")
        print(f"  std ratio b973c/gxtb: {blk['std_ratio_b973c_over_gxtb']}  "
              f"(pilot same-pairs one-shot geom: {PILOT_B973C_STD / PILOT_GXTB_STD:.2f})")

    ov = out["groups"]["overall"]
    ratio = ov["std_ratio_b973c_over_gxtb"]
    rb = ov.get("resid_b973c", {}).get("std")
    rg = ov.get("resid_gxtb", {}).get("std")
    repro = ov.get("repro_r2scan", {}).get("std")
    notes = []
    if repro is not None and repro > 3.5:
        notes.append(f"repro_r2scan std {repro} > 3.5: funnel_v3 regen not reproducing stored "
                     f"labels well -> residuals partly geometry-regen noise; read ratio cautiously.")
    if ratio is None:
        verdict = "INCONCLUSIVE: could not compute std ratio."
    elif ratio <= 0.45 and rb is not None and rb <= 2.5:
        verdict = (f"GREEN: B97-3c keeps low scatter at full scale (std {rb} vs g-xTB {rg}, "
                   f"ratio {ratio}). Proceed with BASELINE_COL=dG_b973c_kcal champion+GNN retrain.")
    elif ratio <= 0.7:
        verdict = (f"AMBER: B97-3c still helps (ratio {ratio}, std {rb} vs {rg}) but less than "
                   f"the pilot's 0.26. Retrain still worth it; expect a smaller floor drop.")
    else:
        verdict = (f"RED: B97-3c advantage largely gone at full scale (ratio {ratio}, std {rb} "
                   f"vs {rg}). Reconsider committing the baseline swap.")
    out["verdict"] = verdict
    out["notes"] = notes
    print(f"\n>>> {verdict}")
    for n in notes:
        print(f"  note: {n}")
    Path(f"{args.out_prefix}_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_prefix}_merged.csv / _labels.csv / _summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
