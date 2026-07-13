"""Deep visualization of aldehydes_rejected_v5.csv — what the v5 filter REMOVED.

Companion to analyze_benzoin_deep.py (which profiles the KEPT set). Here the primary
axis is `reject_reason`: how much each (changed) filter criterion removed, the property
distributions of the discarded molecules, and which rejections are hard/format-based vs
chemically-motivated benzoin exclusions (so borderline/recoverable buckets are visible).

v5 = v4 with the `vinyl_conj` CHO class now rejected outright (a new benzoin-chemistry
reject_reason alongside enal/ynal), so the kept set is purely aromatic + aliphatic.

Reads:  data/library/aldehydes_rejected_v5.csv  (+ clean_v5 MW for a kept overlay)
Writes: data/analysis/aldehyde_v5_rejected/*.png
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
SRC  = REPO / "data/library/aldehydes_rejected_v5.csv"
KEPT = REPO / "data/library/aldehydes_clean_v5.csv"
OUT  = REPO / "data/analysis/aldehyde_v5_rejected"
OUT.mkdir(parents=True, exist_ok=True)

C = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3","#8C8C8C"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# Group the reject reasons into interpretable families.
CATEGORY = {
    "multi_component":     "format / hard",
    "net_charged":         "format / hard",
    "invalid_parse":       "format / hard",
    "isotope":             "format / hard",
    "disallowed_element":  "element (exotic / metal)",
    "mw_too_high":         "size (MW > 500)",
    "aliphatic_too_large": "size (aliphatic C > 12)",
    "not_single_aldehyde": "aldehyde-count (0 true CHO)",
    "multi_aldehyde":      "aldehyde-count (> 1 CHO)",
    "enal":                "benzoin chemistry",
    "ynal":                "benzoin chemistry",
    "vinyl_conj":          "benzoin chemistry",
    "alpha_dicarbonyl":    "benzoin chemistry",
    "reactive_group":      "benzoin chemistry",
    "zwitterion_or_ylide": "benzoin chemistry",
}
CAT_COLORS = {
    "format / hard": C[7], "element (exotic / metal)": C[5],
    "size (MW > 500)": C[3], "size (aliphatic C > 12)": C[1],
    "aldehyde-count (0 true CHO)": C[4], "aldehyde-count (> 1 CHO)": C[6],
    "benzoin chemistry": C[2],
}

# ── Single pass ────────────────────────────────────────────────────────────────
print(f"Reading {SRC.name} ...")
reason_l, mw_l, nc_l, rings_l, hba_l, rot_l, logp_l, arom_l = [], [], [], [], [], [], [], []
disallowed_elems = Counter()       # elements responsible for disallowed_element rejects
ALLOWED_Z = {1,6,7,8,9,5,14,15,16,17,34,35,53}

total = 0
with open(SRC) as f:
    for row in csv.DictReader(f):
        reason = row["reject_reason"]
        mol = Chem.MolFromSmiles(row["SMILES"])
        total += 1
        reason_l.append(reason)
        try:
            mw = float(row["MW"]) if row.get("MW") else (Descriptors.MolWt(mol) if mol else np.nan)
        except ValueError:
            mw = np.nan
        mw_l.append(mw)
        if mol is None:
            nc_l.append(np.nan); rings_l.append(np.nan); hba_l.append(np.nan)
            rot_l.append(np.nan); logp_l.append(np.nan); arom_l.append(False)
            continue
        nc_l.append(sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C"))
        rings_l.append(rdMolDescriptors.CalcNumRings(mol))
        hba_l.append(rdMolDescriptors.CalcNumHBA(mol))
        rot_l.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        try:
            logp_l.append(Descriptors.MolLogP(mol))
        except Exception:
            logp_l.append(np.nan)
        arom_l.append(rdMolDescriptors.CalcNumAromaticRings(mol) >= 1)
        if reason == "disallowed_element":
            pt = Chem.GetPeriodicTable()
            for z in {a.GetAtomicNum() for a in mol.GetAtoms()} - ALLOWED_Z:
                disallowed_elems[pt.GetElementSymbol(z)] += 1
        if total % 50000 == 0:
            print(f"  {total:,} done...")

print(f"Total rejected: {total:,}")
reason = np.array(reason_l)
mw   = np.array(mw_l, dtype=float); nc = np.array(nc_l, dtype=float)
rings= np.array(rings_l, dtype=float); hba = np.array(hba_l, dtype=float)
rot  = np.array(rot_l, dtype=float);  logp = np.array(logp_l, dtype=float)
arom = np.array(arom_l)
rc = pd.Series(reason).value_counts()
n_kept = sum(1 for _ in open(KEPT)) - 1
n_raw  = total + n_kept

def fmt(ax): ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))

# ════════════════════════════════════════════════════════════════════════════
# Fig 1: Reject-reason overview (counts, % of raw, and family grouping)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"v4 Rejections — {total:,} of {n_raw:,} raw removed "
             f"({total/n_raw*100:.0f}%); {n_kept:,} kept", fontsize=13, fontweight="bold")

ax = axes[0]
colors_r = [CAT_COLORS[CATEGORY.get(r, "format / hard")] for r in rc.index]
bars = ax.barh(range(len(rc)), rc.values, color=colors_r, alpha=0.88)
ax.set_yticks(range(len(rc))); ax.set_yticklabels(rc.index, fontsize=9)
ax.invert_yaxis(); ax.set_xlabel("molecules removed")
ax.set_title("By reject_reason (coloured by family)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}k"))
for i, v in enumerate(rc.values):
    ax.text(v+400, i, f"{v:,} ({v/n_raw*100:.1f}%)", va="center", fontsize=7.5)

# family pie
ax = axes[1]
cat_counts = Counter()
for r, n in rc.items():
    cat_counts[CATEGORY.get(r, "format / hard")] += n
cats = sorted(cat_counts, key=lambda k: -cat_counts[k])
ax.pie([cat_counts[k] for k in cats],
       labels=[f"{k}\n{cat_counts[k]:,} ({cat_counts[k]/total*100:.0f}%)" for k in cats],
       colors=[CAT_COLORS[k] for k in cats], startangle=90, wedgeprops=dict(width=0.45),
       textprops={"fontsize": 8})
ax.set_title("Rejection families")
fig.tight_layout(); fig.savefig(OUT/"01_reject_reasons.png"); plt.close(fig)
print("Saved: 01_reject_reasons.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 2: MW of rejected — overall, the >500 tail, kept overlay, MW by reason
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Molecular Weight of rejected molecules", fontsize=13, fontweight="bold")

ax = axes[0,0]
finite = mw[np.isfinite(mw)]
ax.hist(np.clip(finite, 0, 1200), bins=120, color=C[3], alpha=0.85, edgecolor="none")
ax.axvline(500, ls="--", color="black", lw=1, label="MW=500 cap")
ax.set_xlabel("MW (Da, clipped @1200)"); ax.set_ylabel("Count")
ax.set_title(f"All rejected MW (max={np.nanmax(mw):,.0f} → polymer junk)"); ax.legend(); fmt(ax)

ax = axes[0,1]
mh = mw[reason == "mw_too_high"]
band = mh[(mh > 500) & (mh <= 800)]
ax.hist(band, bins=60, color=C[1], alpha=0.85, edgecolor="none")
ax.set_xlabel("MW (Da)"); ax.set_ylabel("Count")
just_over = ((mh > 500) & (mh <= 550)).sum()
ax.set_title(f"mw_too_high tail 500–800 Da\n({just_over:,} are 500–550, borderline-recoverable)")
fmt(ax)

ax = axes[1,0]   # rejected (≤1200) vs kept MW overlay
kept_mw = pd.read_csv(KEPT, usecols=["MW"])["MW"].to_numpy()
ax.hist(kept_mw, bins=80, color=C[2], alpha=0.6, density=True, label=f"kept ({n_kept:,})", edgecolor="none")
ax.hist(np.clip(finite,0,1200), bins=80, color=C[3], alpha=0.6, density=True,
        label=f"rejected ({total:,})", edgecolor="none")
ax.axvline(500, ls="--", color="black", lw=1)
ax.set_xlabel("MW (Da)"); ax.set_ylabel("Density"); ax.set_title("Kept vs Rejected MW"); ax.legend()

ax = axes[1,1]   # MW boxplot by top reasons
top_reasons = [r for r in rc.index if r != "mw_too_high"][:7]
data_box = [mw[(reason == r) & np.isfinite(mw)] for r in top_reasons]
data_box = [np.clip(d, 0, 800) for d in data_box]
bp = ax.boxplot(data_box, vert=False, patch_artist=True, showfliers=False,
                medianprops=dict(color="black"))
for patch, r in zip(bp["boxes"], top_reasons):
    patch.set_facecolor(CAT_COLORS[CATEGORY.get(r, "format / hard")]); patch.set_alpha(0.7)
ax.set_yticklabels(top_reasons, fontsize=8); ax.set_xlabel("MW (Da, clipped @800)")
ax.set_title("MW by reason (excl. mw_too_high)")
fig.tight_layout(); fig.savefig(OUT/"02_mw_rejected.png"); plt.close(fig)
print("Saved: 02_mw_rejected.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 3: Property distributions for the major reject reasons
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Properties of rejected molecules (top reasons)", fontsize=13, fontweight="bold")
show = list(rc.index[:6])
for data, label, ax, mx in [
    (nc,   "Carbon Count",     axes[0,0], 40),
    (rings,"Ring Count",       axes[0,1], 8),
    (logp, "LogP",             axes[1,0], None),
    (rot,  "Rotatable Bonds",  axes[1,1], 20),
]:
    for r in show:
        d = data[(reason == r) & np.isfinite(data)]
        if len(d) < 50: continue
        if mx: d = np.clip(d, 0, mx)
        ax.hist(d, bins=40, histtype="step", lw=1.6, density=True,
                color=CAT_COLORS[CATEGORY.get(r,"format / hard")], label=f"{r} ({len(d):,})")
    ax.set_xlabel(label); ax.set_ylabel("Density"); ax.set_title(label); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(OUT/"03_properties_by_reason.png"); plt.close(fig)
print("Saved: 03_properties_by_reason.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 4: disallowed_element breakdown + benzoin-chemistry exclusions detail
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Element rejects & benzoin-chemistry exclusions", fontsize=13, fontweight="bold")

ax = axes[0]
de = disallowed_elems.most_common(15)
if de:
    els, cnts = zip(*de)
    ax.barh(range(len(els)), cnts, color=C[5], alpha=0.85)
    ax.set_yticks(range(len(els))); ax.set_yticklabels(els); ax.invert_yaxis()
    ax.set_xlabel("count"); ax.set_title(f"disallowed_element — exotic/metal ({rc.get('disallowed_element',0):,})")
    for i, v in enumerate(cnts):
        ax.text(v+0.5, i, f"{v}", va="center", fontsize=8)

ax = axes[1]
chem = ["enal", "multi_aldehyde", "alpha_dicarbonyl", "ynal", "vinyl_conj", "reactive_group", "zwitterion_or_ylide"]
chem = [r for r in chem if r in rc.index]
tot_c = [rc[r] for r in chem]
arom_c = [int(((reason == r) & arom).sum()) for r in chem]
y = range(len(chem))
ax.barh(y, tot_c, color=C[2], alpha=0.5, label="all")
ax.barh(y, arom_c, color=C[0], alpha=0.9, label="of which aromatic-ring")
ax.set_yticks(list(y)); ax.set_yticklabels(chem, fontsize=9); ax.invert_yaxis()
ax.set_xlabel("count"); ax.set_title("Benzoin-chemistry exclusions (aromatic share)"); ax.legend(fontsize=8)
for i, (t, a) in enumerate(zip(tot_c, arom_c)):
    ax.text(t+200, i, f"{t:,} ({a/t*100:.0f}% ar)", va="center", fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT/"04_elements_chemistry.png"); plt.close(fig)
print("Saved: 04_elements_chemistry.png")

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Rejected total      : {total:,} ({total/n_raw*100:.1f}% of {n_raw:,} raw)")
print("By family:")
for k in cats:
    print(f"  {k:32s}: {cat_counts[k]:8,d} ({cat_counts[k]/total*100:4.1f}%)")
print(f"MW>500 (mw_too_high): {rc.get('mw_too_high',0):,}; of which 500–550 Da: "
      f"{int(((mw[reason=='mw_too_high']>500)&(mw[reason=='mw_too_high']<=550)).sum()):,} (borderline)")
print(f"disallowed_element  : {rc.get('disallowed_element',0):,}  top elems: "
      f"{', '.join(f'{e}({n})' for e,n in disallowed_elems.most_common(6))}")
print(f"Benzoin-chem excl.  : enal {rc.get('enal',0):,}, multi_ald {rc.get('multi_aldehyde',0):,}, "
      f"α-dicarbonyl {rc.get('alpha_dicarbonyl',0):,}, ynal {rc.get('ynal',0):,}, "
      f"vinyl_conj {rc.get('vinyl_conj',0):,}, reactive {rc.get('reactive_group',0):,}")
print(f"\nAll plots → {OUT}")
