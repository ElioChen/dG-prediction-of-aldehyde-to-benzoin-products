#!/usr/bin/env python3
"""
Diff the funnel_v3 re-label against the current production labels: which molecules'
dG_orca_kcal changed when the topology guard removed broken-connectivity conformers,
and by how much. Runnable while the re-label array is still going (globs whatever is
done). The expectation (from the 80-mol A/B): ~3% of molecules move, the rest are
within numerical noise.

Usage
  python pipeline/relabel_diff.py
  python pipeline/relabel_diff.py --new-dir data/raw/featurize_funnelv3_relabel \
                                  --old data/featurize.parquet --thresh 1.0
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def load_new(d: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(f"{d}/mol_*/features.csv")):
        try:
            r = pd.read_csv(f)
            if len(r):
                rows.append(r.iloc[0])
        except Exception:
            pass
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", default=str(REPO / "data/raw/featurize_funnelv3_relabel"))
    ap.add_argument("--old", default=str(REPO / "data/featurize.parquet"))
    ap.add_argument("--thresh", type=float, default=1.0, help="|Δ dG_orca| to flag (kcal)")
    a = ap.parse_args()

    new = load_new(a.new_dir)
    old = pd.read_parquet(a.old)[["index", "SMILES", "dG_orca_kcal"]]
    print(f"re-labeled so far: {len(new)} / {len(old)} production molecules")
    if new.empty:
        print("…nothing done yet."); return 0

    new_err = new["error"].notna().sum() if "error" in new else 0
    print(f"  re-label errors: {new_err}")
    m = old.merge(new[["index", "dG_orca_kcal"]], on="index",
                  suffixes=("_old", "_v3"))
    both = m[m["dG_orca_kcal_old"].notna() & m["dG_orca_kcal_v3"].notna()].copy()
    both["delta"] = both["dG_orca_kcal_v3"] - both["dG_orca_kcal_old"]
    d = both["delta"].to_numpy()
    print(f"\nΔ(dG_orca) = v3 − old, n={len(both)}:")
    print(f"  mean {d.mean():+.3f}  std {d.std():.3f}  MAD {np.abs(d).mean():.3f}  "
          f"max|Δ| {np.abs(d).max():.2f}")
    for t in (1.0, 2.0, 5.0, 10.0):
        print(f"  |Δ| > {t:>4.0f} kcal: {(np.abs(d) > t).sum()} "
              f"({100*(np.abs(d) > t).mean():.1f}%)")

    flagged = both[np.abs(both["delta"]) > a.thresh].sort_values(
        "delta", key=lambda s: s.abs(), ascending=False)
    print(f"\n{len(flagged)} molecules changed |Δ| > {a.thresh} kcal "
          f"(the broken-topology fixes):")
    for _, r in flagged.head(40).iterrows():
        print(f"  idx {int(r['index']):>6}  Δ={r['delta']:+7.2f}  "
              f"old={r['dG_orca_kcal_old']:7.2f} v3={r['dG_orca_kcal_v3']:7.2f}  "
              f"{r['SMILES']}")
    flagged.to_csv(REPO / "runs/relabel_v3_changes.csv", index=False)
    print(f"\nwrote runs/relabel_v3_changes.csv  "
          f"(when complete: assemble_featurize.py --globs "
          f"'{a.new_dir}/mol_*/features.csv' --out data/featurize_funnelv3.parquet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
