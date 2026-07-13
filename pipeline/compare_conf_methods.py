#!/usr/bin/env python3
"""
Compare two conformer-engine featurize runs over subset_conftest_v2 (funnel_v2 vs
CREST). Reports, per molecule, the agreement of the DFT label dG_orca_kcal (and the
xTB dG_xtb_kcal) and flags the large disagreements — where the conformer SEARCH, not
the DFT, drove the label. Runnable while the arrays are still going (skips missing).

The earlier diagnostic showed the funnel can return a BROKEN-connectivity lowest
conformer whose spurious energy poisons the Boltzmann label; CREST's MTD + topology
check cannot. So a clean win looks like: the two methods agree on most molecules and
disagree on a few, and the flagged few are funnel failures (verify with
diag_conf_connectivity-style topology checks on the flagged subset).

Usage
  python pipeline/compare_conf_methods.py
  python pipeline/compare_conf_methods.py --a <dirA> --b <dirB> --thresh 2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def load_run(d: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(d.glob("mol_*/features.csv")):
        try:
            r = pd.read_csv(f)
            if len(r):
                rows.append(r.iloc[0])
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(REPO / "data/raw/featurize_conftest2_funnelv2"))
    ap.add_argument("--b", default=str(REPO / "data/raw/featurize_conftest2_crest"))
    ap.add_argument("--meta", default=str(REPO / "data/library/subset_conftest_v2.csv"))
    ap.add_argument("--thresh", type=float, default=2.0, help="|Δ dG_orca| flag (kcal)")
    ap.add_argument("--out", default=str(REPO / "runs/conf_method_compare.csv"))
    args = ap.parse_args()

    A, B = load_run(Path(args.a)), load_run(Path(args.b))
    print(f"funnel_v2 ({Path(args.a).name}): {len(A)} done")
    print(f"crest     ({Path(args.b).name}): {len(B)} done")
    if A.empty or B.empty:
        print("…not enough results yet."); return 0

    for tag, df in (("funnel_v2", A), ("crest", B)):
        nerr = df["error"].notna().sum() if "error" in df else 0
        print(f"  {tag}: errors={nerr}")

    keep = ["index", "dG_xtb_kcal", "dG_orca_kcal", "error"]
    m = A[keep].merge(B[keep], on="index", suffixes=("_fun", "_crest"))
    meta = pd.read_csv(args.meta)[["index", "SMILES", "cho_class",
                                   "product_rotbonds", "flex_bin"]]
    m = m.merge(meta, on="index", how="left")

    both = m[m["dG_orca_kcal_fun"].notna() & m["dG_orca_kcal_crest"].notna()].copy()
    print(f"\nboth have dG_orca: {len(both)}")
    if both.empty:
        # fall back to xTB-level agreement if DFT not in yet
        both = m[m["dG_xtb_kcal_fun"].notna() & m["dG_xtb_kcal_crest"].notna()].copy()
        if both.empty:
            print("no overlapping labels yet."); return 0
        col_fun, col_cr, lbl = "dG_xtb_kcal_fun", "dG_xtb_kcal_crest", "dG_xtb"
        print(f"(DFT not ready — comparing xTB labels, n={len(both)})")
    else:
        col_fun, col_cr, lbl = "dG_orca_kcal_fun", "dG_orca_kcal_crest", "dG_orca"

    both["delta"] = both[col_cr] - both[col_fun]
    d = both["delta"].to_numpy()
    print(f"\nΔ({lbl}) = crest − funnel_v2  (kcal/mol), n={len(both)}")
    print(f"  mean {d.mean():+.2f}  std {d.std():.2f}  MAD {np.abs(d).mean():.2f}  "
          f"max|Δ| {np.abs(d).max():.2f}")
    for t in (2.0, 5.0, 10.0):
        print(f"  |Δ| > {t:>4.0f} kcal: {(np.abs(d) > t).sum()}")
    print("\nby flex_bin (mean |Δ|):")
    print(both.assign(absd=np.abs(both["delta"]))
          .groupby("flex_bin")["absd"].agg(["count", "mean", "max"]).to_string())

    flagged = both[np.abs(both["delta"]) > args.thresh].sort_values(
        "delta", key=lambda s: s.abs(), ascending=False)
    print(f"\n{len(flagged)} disagreements |Δ| > {args.thresh} kcal "
          f"(search-driven — inspect topology of the lowest conformer):")
    for _, r in flagged.head(20).iterrows():
        print(f"  idx {int(r['index']):>6}  Δ={r['delta']:+6.2f}  "
              f"fun={r[col_fun]:7.2f} crest={r[col_cr]:7.2f}  "
              f"rb={int(r['product_rotbonds'])} {r['flex_bin']:11s} {r['cho_class']}")
        print(f"            {r['SMILES']}")

    both.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
