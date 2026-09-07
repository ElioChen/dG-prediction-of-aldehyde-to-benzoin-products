#!/usr/bin/env python3
"""Build the Rec-1 production-geometry recheck pair list from the D-pilot holdout.

Default: ALL 128 pilot pairs (queue permitting) so the recheck is directly comparable
to the pilot headline resid stds (g-xTB 4.32 / B97-3c 1.11 on 128 pairs).
--n-per-grp N  -> instead take N per group, stratified across the pilot resid_b973c
range (use for a fast subset, e.g. 15).

Output: data/cross_benzoin/rec1_prodgeom_recheck/rec1_recheck_pairs<K>.csv
"""
import argparse
import csv
import statistics as st
from pathlib import Path

from rdkit import Chem

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
SAMPLE = REPO / "data/cross_benzoin/cheap_baseline_pilot/sample.csv"
MERGED = REPO / "data/cross_benzoin/cheap_baseline_pilot/cheap_baseline_pilot_merged.csv"
OUTDIR = REPO / "data/cross_benzoin/rec1_prodgeom_recheck"


def heavy(smi):
    m = Chem.MolFromSmiles(str(smi))
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() != "H") if m else 10**9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-grp", type=int, default=0, help="0 = all pairs; else N/group stratified")
    ap.add_argument("--max-heavy", type=int, default=100, help="product heavy-atom cap (safety net)")
    a = ap.parse_args()

    srt = {r["id"]: r for r in csv.DictReader(open(SAMPLE, encoding="utf-8-sig"))}
    mrg = {r["id"]: r for r in csv.DictReader(open(MERGED, encoding="utf-8-sig"))}

    pool = {"hetero_hardtail": [], "control": []}
    for pid, m in mrg.items():
        s = srt.get(pid)
        if not s:
            continue
        try:
            rb = float(m["resid_b973c"]); rg = float(m["resid_gxtb"]); lbl = float(m["dG_orca_kcal_stored"])
        except (ValueError, KeyError):
            continue
        h = heavy(s["smiles"])
        if h > a.max_heavy:
            continue
        if Chem.MolFromSmiles(s["donor_smiles"]) is None or Chem.MolFromSmiles(s["acceptor_smiles"]) is None:
            continue
        pool.setdefault(m["grp"], []).append(dict(
            pid=pid, grp=m["grp"], donor_smiles=s["donor_smiles"], acceptor_smiles=s["acceptor_smiles"],
            prod_smiles=s["smiles"], label=lbl, gxtb_stored=m.get("dG_gxtb_kcal_stored", ""),
            pilot_resid_gxtb=rg, pilot_resid_b973c=rb, prod_heavy=h))

    picked = []
    for grp, rows in pool.items():
        rows.sort(key=lambda d: d["pilot_resid_b973c"])
        if a.n_per_grp <= 0 or a.n_per_grp >= len(rows):
            sel = rows
        else:
            n = len(rows)
            idxs = sorted({round(i * (n - 1) / (a.n_per_grp - 1)) for i in range(a.n_per_grp)})
            j = 0
            while len(idxs) < a.n_per_grp and j < n:
                if j not in idxs:
                    idxs.append(j)
                j += 1
            sel = [rows[i] for i in sorted(idxs)[:a.n_per_grp]]
        picked += sel
        print(f"{grp}: pool={len(rows)} picked={len(sel)} "
              f"resid_b973c [{sel[0]['pilot_resid_b973c']:+.2f},{sel[-1]['pilot_resid_b973c']:+.2f}] "
              f"heavy [{min(r['prod_heavy'] for r in sel)},{max(r['prod_heavy'] for r in sel)}]")

    out = OUTDIR / f"rec1_recheck_pairs{len(picked)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pid", "grp", "donor_smiles", "acceptor_smiles", "prod_smiles", "label", "gxtb_stored",
            "pilot_resid_gxtb", "pilot_resid_b973c", "prod_heavy"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in picked:
            w.writerow({k: r[k] for k in cols})
    print(f"\nwrote {len(picked)} pairs -> {out}")
    for grp in sorted({r["grp"] for r in picked}):
        sub = [r for r in picked if r["grp"] == grp]
        print(f"  {grp}: pilot resid_gxtb std {st.pstdev([r['pilot_resid_gxtb'] for r in sub]):.2f}  "
              f"resid_b973c std {st.pstdev([r['pilot_resid_b973c'] for r in sub]):.2f}")
    print(f"  ALL: pilot resid_gxtb std {st.pstdev([r['pilot_resid_gxtb'] for r in picked]):.2f}  "
          f"resid_b973c std {st.pstdev([r['pilot_resid_b973c'] for r in picked]):.2f}  (pilot ref 4.32 / 1.11)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
