#!/usr/bin/env python
"""Merge + QC the homo SP-on-archived-geom relabel shards.

`homo_sp_from_geom_worker.py` writes one row per pair to
relabel_sp/shards/shard_%05d.csv (append+fsync). This merges them, keyed on
`id`, reports coverage vs the 184,052-row manifest, and prints the QC that
matters here:

  * repro_r2scan (= new r2SCAN-3c dG - stored 30k label) on the ~24.5k
    label-carrying rows -- should be a tight distribution around ~0 (the
    smoke gave +0.06 / +8e-6 / +2.9); std well under the ~2.9 kcal
    single-conformer label-noise floor = SPs are clean, no protocol offset.
  * resid_b973c (= dG_r2scan - dG_b973c) -- the Delta-learning baseline gap
    for a B97-3c-baselined homo model; std is the Delta-model MAE floor.

Outputs:
  <prefix>_merged.csv   one row per id (real result preferred)
  <prefix>_labels.csv   id,dG_r2scan_kcal,dG_b973c_kcal  (assembler input)
  <prefix>_summary.json  coverage + per-split repro/resid stats + verdict

Run any time for progress QC; run at drain for the real merge (Phase 2).
"""
from __future__ import annotations
import argparse
import glob
import json
import statistics as st
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
D = REPO / "data/cross_benzoin/homo_standalone/relabel_sp"
MANIFEST = D / "homo_sp_manifest.csv"
NOISE_FLOOR = 2.9   # single-conformer r2SCAN-3c dG label-noise floor (confnoise_cross)


def _s(xs):
    xs = [float(x) for x in xs if pd.notna(x)]
    if not xs:
        return dict(n=0)
    a = sorted(abs(x) for x in xs)
    return dict(n=len(xs), mean=round(st.mean(xs), 3), mean_abs=round(st.mean(a), 3),
                std=round(st.pstdev(xs), 3), median_abs=round(a[len(a) // 2], 3),
                p90_abs=round(a[max(0, int(0.9 * len(a)) - 1)], 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-glob", default=str(D / "shards/shard_*.csv"))
    ap.add_argument("--out-prefix", default=str(D / "homo_sp"))
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    files = sorted(glob.glob(args.shards_glob))
    if not files:
        print("no shard csvs yet")
        return 1
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, dtype={"id": str}))
        except Exception as e:
            parts.append(pd.read_csv(f, dtype={"id": str}, on_bad_lines="skip"))
            print(f"  warn: {Path(f).name} on_bad_lines=skip ({e})")
    df = pd.concat(parts, ignore_index=True)
    df["id"] = df["id"].astype(str).str.strip()
    for c in ("dG_r2scan_kcal", "dG_b973c_kcal", "label_stored", "repro_r2scan",
              "E_prod_r2scan", "E_ald_r2scan", "E_prod_b973c", "E_ald_b973c"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["resid_b973c"] = df["dG_r2scan_kcal"] - df["dG_b973c_kcal"]

    df["_has"] = df["dG_r2scan_kcal"].notna().astype(int)
    df = (df.sort_values(["id", "_has"]).drop_duplicates("id", keep="last")
            .drop(columns="_has").reset_index(drop=True))

    err = df["error"].astype(str).str.strip()
    is_err = err.ne("") & err.ne("nan")
    ok = df[df["dG_r2scan_kcal"].notna() & ~is_err].copy()
    true_fail = df[df["dG_r2scan_kcal"].isna()]
    df.to_csv(f"{args.out_prefix}_merged.csv", index=False)
    ok[["id", "dG_r2scan_kcal", "dG_b973c_kcal"]].to_csv(f"{args.out_prefix}_labels.csv", index=False)

    man = pd.read_csv(args.manifest, dtype={"id": str}, usecols=["id"])
    mids = set(man["id"].astype(str).str.strip())
    got = set(ok["id"])
    cov = len(got & mids)
    print(f"{len(files)} shards -> {len(df)} unique id, {len(ok)} usable, "
          f"{len(true_fail)} true-fail, {int(is_err.sum())} error-tagged")
    print(f"coverage vs manifest ({len(mids)}): {cov} ({100*cov/len(mids):.1f}%)")
    for _, r in true_fail.head(15).iterrows():
        print(f"  FAIL {r['id']}: {r.get('error')!r}")

    qc = ok[ok["repro_r2scan"].notna()]
    out = {"n_shards": len(files), "n_unique_id": len(df), "n_ok": len(ok),
           "n_true_fail": int(len(true_fail)), "n_error_tagged": int(is_err.sum()),
           "manifest_n": len(mids), "coverage": cov,
           "coverage_pct": round(100 * cov / len(mids), 2),
           "qc_repro_r2scan_vs_stored_30k": _s(qc["repro_r2scan"]) if len(qc) else {"n": 0},
           "groups": {}}
    blocks = [("overall", ok)]
    if "new_scaffold_split" in ok.columns:
        blocks += [(g, ok[ok["new_scaffold_split"] == g])
                   for g in sorted(ok["new_scaffold_split"].dropna().unique())]
    for name, sub in blocks:
        out["groups"][name] = {k: _s(sub[k]) for k in ("resid_b973c", "repro_r2scan")
                               if k in sub.columns}
        print(f"\n=== {name} (n={len(sub)}) ===")
        for k in ("resid_b973c", "repro_r2scan"):
            if k in out["groups"][name]:
                print(f"  {k:13s}: {out['groups'][name][k]}")

    repro_std = out["qc_repro_r2scan_vs_stored_30k"].get("std")
    repro_mean = out["qc_repro_r2scan_vs_stored_30k"].get("mean")
    if repro_std is None:
        verdict = "INCONCLUSIVE: no label-carrying rows merged yet."
    elif abs(repro_mean or 0) <= 0.5 and repro_std <= NOISE_FLOOR:
        verdict = (f"GREEN: new r2SCAN-3c SPs reproduce the stored 30k labels "
                   f"(repro mean {repro_mean}, std {repro_std} <= {NOISE_FLOOR} floor). "
                   f"No protocol offset. Proceed to assemble + train.")
    elif repro_std <= NOISE_FLOOR + 1.0:
        verdict = (f"AMBER: repro std {repro_std} (mean {repro_mean}) near the noise floor; "
                   f"usable, watch the offset.")
    else:
        verdict = (f"RED: repro std {repro_std} (mean {repro_mean}) too wide -- geometry or "
                   f"SP inconsistency. Investigate before training.")
    out["verdict"] = verdict
    print(f"\n>>> {verdict}")
    Path(f"{args.out_prefix}_summary.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out_prefix}_merged.csv / _labels.csv / _summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
