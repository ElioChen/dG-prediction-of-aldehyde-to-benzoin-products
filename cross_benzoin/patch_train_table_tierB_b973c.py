#!/usr/bin/env python3
"""Drain-time: fold the Tier B self-consistent relabel into the champion training table.

Takes the current frozen slim260 r1-N table and the merged Tier B labels, and writes a
slim257 table carrying the NEW self-consistent labels + the B97-3c baseline, ready for
`train_scaffold_disjoint.py` (run it with CB_TARGET_COL=dG_r2scan_kcal
CB_BASELINE_COL=dG_b973c_kcal) and the GNN 4-seed retrain.  HANDOFF sec 1.3 step 2.

What it does, keyed on id (== Tier B `pid`, verified 35,528/35,528 exact, string inchikeys
so no norm_id needed here):
  * join dG_r2scan_kcal / dG_b973c_kcal / dG_gxtb_kcal(new, same-geom) from the merge
  * TARGET  : dG_r2scan_kcal  (new self-consistent project label; replaces old dG_orca_kcal,
              which is kept as dG_orca_kcal_stored for provenance/repro diagnostics)
  * BASELINE: dG_b973c_kcal   (cheap self-consistent baseline; the lever)
  * also keeps dG_gxtb_kcal_selfconsistent for a same-geometry A/B
  * drops the 3 degenerate n_CHO features -> 257 schema v2
  * optional winsorise of the heavy-tailed features flagged in
    docs/feature_audit_20260908.md (--winsor)
  * --keep-unrelabeled {drop,old}: rows with no Tier B label yet -> dropped (default) or
    fall back to the old dG_orca_kcal label (NOT recommended; mixes label levels)

No-op self-test: --labels <(old dG_orca as dG_r2scan)> reproduces the slim257 table so the
downstream plumbing can be validated against the known champion numbers before drain.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEF_TABLE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
DEF_LABELS = REPO / "data/cross_benzoin/rec1_b973c_tierB/rec1_b973c_tierB_labels.csv"
DEF_F257 = REPO / "data/cross_benzoin/feature_list_257_no_nCHO_v2.json"
NCHO = ["donor_n_CHO", "acceptor_n_CHO", "product_n_CHO"]
# Heavy-tail features the 2026-09-08 audit named explicitly (a handful of outlier
# rows each, pathological funnel_v3 geoms). The audit says: re-run the audit AFTER
# the Tier B geometry regen and winsorise at [p1,p99] only if the tails persist for
# the same pairs. product_mordred_RPCS is deliberately NOT here (audit: expected
# heavy tail for that CPSA descriptor, not a bug). Match by exact suffix so the
# donor_/acceptor_/product_ prefixed variants are all caught.
WINSOR_SUFFIX = ["wbo_CC_new", "mulliken_carbC", "mulliken_CHO_C"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=DEF_TABLE)
    ap.add_argument("--labels", type=Path, default=DEF_LABELS,
                    help="Tier B merge output: pid,dG_r2scan_kcal,dG_b973c_kcal,dG_gxtb_kcal")
    ap.add_argument("--feature-list-257", type=Path, default=DEF_F257)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-unrelabeled", choices=["drop", "old"], default="drop")
    ap.add_argument("--winsor", action="store_true", help="winsorise heavy-tail feats to [p1,p99] of clean-train")
    ap.add_argument("--selftest-noop", action="store_true",
                    help="ignore --labels; use old dG_orca_kcal as the r2scan label and "
                         "dG_gxtb_kcal as the b973c baseline (reproduces slim257 for plumbing checks)")
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    n0 = len(df)
    if "id" not in df.columns:
        print("ERROR: table has no `id` column", file=sys.stderr); return 1
    df["id"] = df["id"].astype(str).str.strip()

    if args.selftest_noop:
        df["dG_r2scan_kcal"] = df["dG_orca_kcal"].to_numpy()
        df["dG_b973c_kcal"] = df["dG_gxtb_kcal"].to_numpy()
        df["dG_gxtb_kcal_selfconsistent"] = df["dG_gxtb_kcal"].to_numpy()
        matched = np.ones(len(df), dtype=bool)
        print("[selftest-noop] using old dG_orca_kcal as label, dG_gxtb_kcal as baseline")
    else:
        lab = pd.read_csv(args.labels)
        lab = lab.rename(columns={"pid": "id"})
        lab["id"] = lab["id"].astype(str).str.strip()
        lab = lab.drop_duplicates("id")
        need = {"id", "dG_r2scan_kcal", "dG_b973c_kcal", "dG_gxtb_kcal"}
        miss = need - set(lab.columns)
        if miss:
            print(f"ERROR: labels file missing {miss}", file=sys.stderr); return 1
        lab = lab.rename(columns={"dG_gxtb_kcal": "dG_gxtb_kcal_selfconsistent"})
        df = df.merge(lab[["id", "dG_r2scan_kcal", "dG_b973c_kcal", "dG_gxtb_kcal_selfconsistent"]],
                      on="id", how="left")
        matched = df["dG_r2scan_kcal"].notna().to_numpy()
        print(f"Tier B labels matched: {matched.sum()}/{len(df)} "
              f"({100*matched.mean():.1f}%)  unmatched -> {args.keep_unrelabeled}")
        by_split = df.assign(_m=matched).groupby("new_scaffold_split")["_m"].agg(["sum", "size"])
        print(by_split.to_string())
        if args.keep_unrelabeled == "drop":
            df = df[matched].reset_index(drop=True)
        else:  # 'old' — fall back to stored dG_orca_kcal, keep g-xTB baseline for those rows
            df.loc[~matched, "dG_r2scan_kcal"] = df.loc[~matched, "dG_orca_kcal"]
            df.loc[~matched, "dG_b973c_kcal"] = df.loc[~matched, "dG_gxtb_kcal"]
            df.loc[~matched, "dG_gxtb_kcal_selfconsistent"] = df.loc[~matched, "dG_gxtb_kcal"]
            print("  WARNING: --keep-unrelabeled old mixes r2SCAN-3c and stored-DFT label levels")

    df = df.rename(columns={"dG_orca_kcal": "dG_orca_kcal_stored"})

    # ---- prune 260 -> 257 (drop degenerate n_CHO) --------------------------------
    f257 = json.loads(Path(args.feature_list_257).read_text())
    f257 = f257 if isinstance(f257, list) else f257["features"]
    present_ncho = [c for c in NCHO if c in df.columns]
    df = df.drop(columns=present_ncho)
    missing = [c for c in f257 if c not in df.columns]
    if missing:
        print(f"ERROR: {len(missing)} of 257 features absent after patch: {missing[:8]}",
              file=sys.stderr); return 1
    print(f"dropped {present_ncho} -> schema v2 (257 features intact)")

    # ---- optional winsorisation of heavy-tail features --------------------------
    if args.winsor:
        tr = df["new_scaffold_split"] == "train"
        wf = [c for c in f257 if any(c == s or c.endswith("_" + s) for s in WINSOR_SUFFIX)]
        clipped = 0
        for c in wf:
            v = pd.to_numeric(df[c], errors="coerce")
            lo, hi = v[tr].quantile(0.01), v[tr].quantile(0.99)
            if pd.notna(lo) and pd.notna(hi) and hi > lo:
                nout = int(((v < lo) | (v > hi)).sum())
                df[c] = v.clip(lo, hi)
                clipped += nout
        print(f"winsorised {len(wf)} heavy-tail feats to clean-train [p1,p99] ({clipped} values clipped)")

    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"  rows {n0} -> {len(df)} | cols {df.shape[1]}")
    print(f"  TARGET_COL=dG_r2scan_kcal  BASELINE_COL=dG_b973c_kcal")
    for c in ("dG_r2scan_kcal", "dG_b973c_kcal", "dG_gxtb_kcal_selfconsistent", "dG_orca_kcal_stored"):
        s = pd.to_numeric(df[c], errors="coerce")
        print(f"  {c:28s} mean {s.mean():7.3f}  std {s.std():6.3f}  na {s.isna().sum()}")
    r = pd.to_numeric(df["dG_r2scan_kcal"], errors="coerce") - pd.to_numeric(df["dG_b973c_kcal"], errors="coerce")
    print(f"  resid (r2scan-b973c)         mean {r.mean():7.3f}  std {r.std():6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
