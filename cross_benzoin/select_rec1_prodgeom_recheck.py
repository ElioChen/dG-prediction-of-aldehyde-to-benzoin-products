#!/usr/bin/env python3
"""Pick 30 D-pilot holdout pairs (15 hetero_hardtail + 15 control) for the production-
geometry recheck. Stratify each group across the pilot resid_b973c quantiles so the
recheck tests whether low B97-3c scatter holds across easy AND hard pairs. Cap product
heavy-atom count to keep ORCA r2SCAN-3c SP tractable. Deterministic (seed 20260908)."""
import csv
from pathlib import Path
from rdkit import Chem

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
SAMPLE = REPO / "data/cross_benzoin/cheap_baseline_pilot/sample.csv"
MERGED = REPO / "data/cross_benzoin/cheap_baseline_pilot/cheap_baseline_pilot_merged.csv"
OUT = REPO / "data/cross_benzoin/rec1_prodgeom_recheck/rec1_recheck_pairs30.csv"
N_PER_GRP = 15
MAX_HEAVY = 70          # product heavy atoms; r2SCAN-3c SP stays ~<1h
SEED = 20260908

srt = {r["id"]: r for r in csv.DictReader(open(SAMPLE, encoding="utf-8-sig"))}
mrg = {r["id"]: r for r in csv.DictReader(open(MERGED, encoding="utf-8-sig"))}

def heavy(smi):
    m = Chem.MolFromSmiles(str(smi))
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() != "H") if m else 10**9

pool = {"hetero_hardtail": [], "control": []}
for pid, m in mrg.items():
    s = srt.get(pid)
    if not s:
        continue
    try:
        rb = float(m["resid_b973c"]); rg = float(m["resid_gxtb"])
        lbl = float(m["dG_orca_kcal_stored"])
    except (ValueError, KeyError):
        continue
    h = heavy(s["smiles"])
    if h > MAX_HEAVY:
        continue
    if Chem.MolFromSmiles(s["donor_smiles"]) is None or Chem.MolFromSmiles(s["acceptor_smiles"]) is None:
        continue
    pool[m["grp"]].append(dict(
        pid=pid, grp=m["grp"], donor_smiles=s["donor_smiles"], acceptor_smiles=s["acceptor_smiles"],
        prod_smiles=s["smiles"], label=lbl, gxtb_stored=m.get("dG_gxtb_kcal_stored", ""),
        pilot_resid_gxtb=rg, pilot_resid_b973c=rb, prod_heavy=h))

picked = []
for grp, rows in pool.items():
    rows.sort(key=lambda d: d["pilot_resid_b973c"])
    n = len(rows)
    # even stratified positions across the sorted-by-resid list
    idxs = sorted({round(i * (n - 1) / (N_PER_GRP - 1)) for i in range(N_PER_GRP)})
    # if collisions reduced the count, backfill deterministically
    j = 0
    while len(idxs) < N_PER_GRP and j < n:
        if j not in idxs:
            idxs.append(j)
        j += 1
    idxs = sorted(idxs)[:N_PER_GRP]
    for i in idxs:
        picked.append(rows[i])
    print(f"{grp}: pool={n}  picked={len(idxs)}  "
          f"resid_b973c range [{rows[0]['pilot_resid_b973c']:+.2f}, {rows[-1]['pilot_resid_b973c']:+.2f}]  "
          f"heavy range [{min(r['prod_heavy'] for r in rows)}, {max(r['prod_heavy'] for r in rows)}]")

OUT.parent.mkdir(parents=True, exist_ok=True)
cols = ["pid", "grp", "donor_smiles", "acceptor_smiles", "prod_smiles", "label", "gxtb_stored",
        "pilot_resid_gxtb", "pilot_resid_b973c", "prod_heavy"]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in picked:
        w.writerow({k: r[k] for k in cols})
print(f"\nwrote {len(picked)} pairs -> {OUT}")
print("\npilot within-geometry std (all 128):  resid_gxtb 4.32  resid_b973c 1.11  (ratio 0.26)")
import statistics as st
for grp in ("hetero_hardtail", "control"):
    sub = [r for r in picked if r["grp"] == grp]
    print(f"  picked {grp}: pilot resid_gxtb std {st.pstdev([r['pilot_resid_gxtb'] for r in sub]):.2f}  "
          f"resid_b973c std {st.pstdev([r['pilot_resid_b973c'] for r in sub]):.2f}")
allp = picked
print(f"  picked ALL30: pilot resid_gxtb std {st.pstdev([r['pilot_resid_gxtb'] for r in allp]):.2f}  "
      f"resid_b973c std {st.pstdev([r['pilot_resid_b973c'] for r in allp]):.2f}")
