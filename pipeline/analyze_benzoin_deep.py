"""Deep visualization analysis of the benzoin aldehyde library.

Adapted for aldehydes_clean_v5.csv (filter_smiles_v5 single-source output). v5 is v4
with the `vinyl_conj` CHO class now REJECTED, so the kept set is purely
{aromatic_carbo, aromatic_hetero, aliphatic}. Key features:

  • reads the v5 clean set on the cluster (was a WSL /mnt/c path);
  • categorizes by the chemically-decisive CHO-environment axis using the `cho_class`
    column (aromatic_carbo / aromatic_hetero / aliphatic) — the single source of truth
    — instead of the crude molecule-level "has aromatic ring" proxy;
  • surfaces the classes the relaxed filter newly RECOVERS: nitro/azide/N-oxide
    (the `xtb_risk` tag) and the B/Si/P/Se elements;
  • adds Fig 9 — the v5 filter funnel (what each criterion removed, incl. the new
    vinyl_conj rejection) alongside the kept-set composition.
"""
import csv
from collections import Counter
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

REPO   = Path(__file__).resolve().parent.parent
SRC    = REPO / "data/library/aldehydes_clean_v5.csv"
REJECT = REPO / "data/library/aldehydes_rejected_v5.csv"
OUT    = REPO / "data/analysis/aldehyde_v5_deep"
OUT.mkdir(parents=True, exist_ok=True)

# Elements the v4 filter newly allows beyond the old {C,H,N,O,F,S,Cl,Br,I} set.
NEW_ELEMENTS = {"B", "Si", "P", "Se"}

# ── Palette ───────────────────────────────────────────────────────────────────
C = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3","#8C8C8C"]
# cho_class → colour (kept consistent across every figure). v5 rejects vinyl_conj,
# so the kept set only ever has these three classes.
CHO_COLORS = {"aromatic_carbo": C[0], "aromatic_hetero": C[1], "aliphatic": C[2]}
CHO_ORDER = ["aromatic_carbo", "aromatic_hetero", "aliphatic"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# ── SMARTS for functional-group prevalence ─────────────────────────────────────
SMARTS = {
    "Nitro":        Chem.MolFromSmarts("[N+](=O)[O-]"),
    "Amine":        Chem.MolFromSmarts("[NX3;H1,H2]"),
    "Tertiary N":   Chem.MolFromSmarts("[NX3;H0;!$(NC=O)]"),
    "Amide":        Chem.MolFromSmarts("[NX3][CX3](=O)"),
    "Ether O":      Chem.MolFromSmarts("[OX2;!$(OC=O)]"),
    "Ester/Acid":   Chem.MolFromSmarts("[CX3](=O)[OX2H0,OX1H1]"),
    "Halogen F":    Chem.MolFromSmarts("[F]"),
    "Halogen Cl":   Chem.MolFromSmarts("[Cl]"),
    "Halogen Br":   Chem.MolFromSmarts("[Br]"),
    "Halogen I":    Chem.MolFromSmarts("[I]"),
    "Hydroxyl":     Chem.MolFromSmarts("[OX2H]"),
    "Ketone":       Chem.MolFromSmarts("[CX3;!H1](=O)[#6]"),
    "Sulfur":       Chem.MolFromSmarts("[S]"),
    "Phosphorus":   Chem.MolFromSmarts("[P]"),
    "Cyano":        Chem.MolFromSmarts("[C]#[N]"),
    "Boronic":      Chem.MolFromSmarts("[#5]"),               # B (newly allowed)
    "Silyl":        Chem.MolFromSmarts("[#14]"),              # Si (newly allowed)
    "Selenium":     Chem.MolFromSmarts("[#34]"),              # Se (newly allowed)
    "Heteroar.":    Chem.MolFromSmarts("a1aaaan1"),
    "Furan/Thio":   Chem.MolFromSmarts("a1aaa[o,s]1"),
}

# ── Single-pass data collection ───────────────────────────────────────────────
print(f"Reading {SRC.name} ...")
mw_list, c_list, ar_list, rot_list, hbd_list, hba_list = [], [], [], [], [], []
rings_list, heteroar_list, tpsa_list, logp_list = [], [], [], []
cho_list, risk_list = [], []
el_counter = Counter()
fg_counter = Counter()

total = 0
with open(SRC) as f:
    for row in csv.DictReader(f):
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        total += 1

        mw   = float(row["MW"]) if row.get("MW") else Descriptors.MolWt(mol)
        cho_list.append(row.get("cho_class", "none"))
        risk_list.append(row.get("xtb_risk", "") or "")

        n_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_c  = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
        mw_list.append(mw); c_list.append(n_c); ar_list.append(n_ar)
        rings_list.append(rdMolDescriptors.CalcNumRings(mol))
        hbd_list.append(rdMolDescriptors.CalcNumHBD(mol))
        hba_list.append(rdMolDescriptors.CalcNumHBA(mol))
        rot_list.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        heteroar_list.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
        tpsa_list.append(rdMolDescriptors.CalcTPSA(mol))
        logp_list.append(Descriptors.MolLogP(mol))

        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in ("C", "H", "O", "N"):
                el_counter[sym] += 1
        for name, pat in SMARTS.items():
            if pat and mol.HasSubstructMatch(pat):
                fg_counter[name] += 1

        if total % 50000 == 0:
            print(f"  {total:,} done...")

print(f"Total processed: {total:,}")

mw   = np.array(mw_list);   c   = np.array(c_list)
ar   = np.array(ar_list);   rings = np.array(rings_list)
hbd  = np.array(hbd_list);  hba = np.array(hba_list)
rot  = np.array(rot_list);  har = np.array(heteroar_list)
tpsa = np.array(tpsa_list); logp = np.array(logp_list)
cho  = np.array(cho_list);  risk = np.array(risk_list)

# cho_class masks (the chemically decisive axis)
m = {k: (cho == k) for k in CHO_ORDER}
is_arom = m["aromatic_carbo"] | m["aromatic_hetero"]   # the project scope

def fmt(ax): ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))

# ════════════════════════════════════════════════════════════════════════════
# Fig 1: MW comprehensive  (now hard-capped ≤500 Da by the v4 filter)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Molecular Weight — {total:,} v4 benzoin aldehydes (MW ≤ 500 Da)",
             fontsize=13, fontweight="bold")

ax = axes[0,0]
ax.hist(mw, bins=120, color=C[0], alpha=0.85, edgecolor="none")
ax.set_xlabel("MW (Da)"); ax.set_ylabel("Count"); ax.set_title("Full distribution"); fmt(ax)

ax = axes[0,1]
for k in CHO_ORDER:
    if m[k].sum():
        ax.hist(mw[m[k]], bins=70, color=CHO_COLORS[k], alpha=0.6,
                label=f"{k} ({m[k].sum():,})", edgecolor="none")
ax.set_xlabel("MW (Da)"); ax.set_ylabel("Count"); ax.set_title("MW by cho_class")
ax.legend(fontsize=7.5); fmt(ax)

ax = axes[1,0]
ax.plot(np.sort(mw), np.arange(1, total+1)/total*100, color=C[0], lw=1.5)
for cut in [200, 300, 400, 500]:
    p = (mw <= cut).mean() * 100
    ax.axvline(cut, ls="--", lw=0.8, color="gray")
    ax.text(cut+2, p-6, f"{cut}Da\n{p:.0f}%", fontsize=7.5, color="gray")
ax.set_xlabel("MW (Da)"); ax.set_ylabel("Cumulative %"); ax.set_title("Cumulative MW")

ax = axes[1,1]
pct_vals = np.percentile(mw, [10,25,50,75,90,95])
ax.barh([f"P{p}" for p in [10,25,50,75,90,95]], pct_vals, color=C[0], alpha=0.8)
for i, v in enumerate(pct_vals):
    ax.text(v+1, i, f"{v:.0f}", va="center", fontsize=9)
ax.set_xlabel("MW (Da)"); ax.set_title("MW percentiles"); ax.set_xlim(0, 560)

fig.tight_layout(); fig.savefig(OUT/"01_mw_analysis.png"); plt.close(fig)
print("Saved: 01_mw_analysis.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 2: Carbon & ring analysis
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Carbon Count & Ring Analysis", fontsize=13, fontweight="bold")

ax = axes[0,0]
vc = pd.Series(c).value_counts().sort_index()
ax.bar(vc.index[:40], vc.values[:40], color=C[0], width=0.85, alpha=0.85)
ax.axvline(np.median(c), ls="--", color="gray", lw=1, label=f"Median: {np.median(c):.0f}")
ax.set_xlabel("Carbon Count"); ax.set_ylabel("Count"); ax.set_title("Carbon count"); ax.legend(); fmt(ax)

for ax, arr, col, title in [
    (axes[0,1], ar,    C[1], "Aromatic ring count"),
    (axes[1,0], rings, C[2], "Total rings (aromatic + aliphatic)"),
    (axes[1,1], har,   C[3], "Heteroaromatic ring count"),
]:
    vcx = pd.Series(arr).value_counts().sort_index()
    ax.bar(vcx.index, vcx.values, color=col, width=0.7, alpha=0.85)
    ax.set_xlabel(title.split("(")[0].strip()); ax.set_ylabel("Count"); ax.set_title(title); fmt(ax)
    for k_, v_ in vcx.items():
        if v_/total > 0.01:
            ax.text(k_, v_+200, f"{v_/total*100:.0f}%", ha="center", fontsize=7)

fig.tight_layout(); fig.savefig(OUT/"02_carbon_rings.png"); plt.close(fig)
print("Saved: 02_carbon_rings.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 3: Drug-likeness properties
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Drug-likeness / Physicochemical Properties", fontsize=13, fontweight="bold")
for ax, data, label, color in [
    (axes[0,0], hbd,  "H-Bond Donors",    C[0]),
    (axes[0,1], hba,  "H-Bond Acceptors", C[1]),
    (axes[0,2], rot,  "Rotatable Bonds",  C[2]),
    (axes[1,0], tpsa, "TPSA (Å²)",        C[3]),
    (axes[1,1], logp, "LogP",             C[4]),
]:
    vc = pd.Series(data).value_counts().sort_index() if data.max() < 30 else None
    if vc is not None and len(vc) <= 25:
        ax.bar(vc.index, vc.values, color=color, width=0.7, alpha=0.85)
    else:
        ax.hist(data, bins=60, color=color, alpha=0.85, edgecolor="none")
    ax.set_xlabel(label); ax.set_ylabel("Count"); ax.set_title(f"{label} (med={np.median(data):.1f})"); fmt(ax)

ax = axes[1,2]
ro5 = ((mw <= 500) & (hbd <= 5) & (hba <= 10) & (logp <= 5)).sum()
ax.pie([ro5, total-ro5],
       labels=[f"Ro5 compliant\n{ro5:,} ({ro5/total*100:.1f}%)",
               f"Ro5 violation\n{total-ro5:,} ({(total-ro5)/total*100:.1f}%)"],
       colors=[C[2], C[0]], startangle=90, wedgeprops=dict(width=0.5))
ax.set_title("Lipinski Rule-of-5")
fig.tight_layout(); fig.savefig(OUT/"03_physicochemical.png"); plt.close(fig)
print("Saved: 03_physicochemical.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 4: Non-CHON elements (B/Si/P/Se newly allowed by v4 — highlighted)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Elemental Composition (non-CHON) — B/Si/P/Se newly allowed in v4 (orange)",
             fontsize=13, fontweight="bold")

top20 = el_counter.most_common(20)
if top20:
    els, cnts = zip(*top20)
    colors_e = [C[1] if e in NEW_ELEMENTS else C[0] for e in els]
    bars = axes[0].barh(range(len(els)), [x/total*100 for x in cnts], color=colors_e, alpha=0.85)
    axes[0].set_yticks(range(len(els))); axes[0].set_yticklabels(els)
    axes[0].set_xlabel("% of compounds"); axes[0].set_title("Non-CHON elements (top 20)")
    axes[0].invert_yaxis()
    for i,(b,c_) in enumerate(zip(bars, cnts)):
        axes[0].text(b.get_width()+0.05, i, f"{c_:,}", va="center", fontsize=7.5)

ax = axes[1]
newel = {e: el_counter.get(e, 0) for e in ["B","Si","P","Se"]}
ax.bar(newel.keys(), [v/total*100 for v in newel.values()], color=C[1], alpha=0.85, width=0.5)
ax.set_ylabel("% of compounds"); ax.set_title("Newly-allowed elements (v4 recovery)")
for i,(el,v) in enumerate(newel.items()):
    ax.text(i, v/total*100+0.003, f"{v:,}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"04_elements.png"); plt.close(fig)
print("Saved: 04_elements.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 5: Functional group prevalence
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
fg_sorted = sorted(fg_counter.items(), key=lambda x: x[1], reverse=True)
fg_names, fg_cnts = zip(*fg_sorted)
bars = ax.barh(range(len(fg_names)), [v/total*100 for v in fg_cnts], color=C[0], alpha=0.85)
ax.set_yticks(range(len(fg_names))); ax.set_yticklabels(fg_names)
ax.set_xlabel("% of compounds containing this group")
ax.set_title(f"Functional Group Prevalence (N={total:,})", fontsize=12, fontweight="bold")
ax.invert_yaxis()
for i,(b,c_) in enumerate(zip(bars, fg_cnts)):
    ax.text(b.get_width()+0.1, i, f"{c_:,} ({c_/total*100:.1f}%)", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(OUT/"05_functional_groups.png", bbox_inches="tight"); plt.close(fig)
print("Saved: 05_functional_groups.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 6: 2D property space
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("2D Property Space", fontsize=13, fontweight="bold")
hb = axes[0].hexbin(mw, logp, gridsize=60, cmap="Blues", bins="log", mincnt=1)
axes[0].set_xlabel("MW (Da)"); axes[0].set_ylabel("LogP"); axes[0].set_title("MW vs LogP (log density)")
plt.colorbar(hb, ax=axes[0], label="log10(count)")
axes[0].axvline(500, ls="--", color="red", lw=0.8, alpha=0.7)
axes[0].axhline(5, ls="--", color="red", lw=0.8, alpha=0.7, label="Ro5 limits"); axes[0].legend(fontsize=8)
hb2 = axes[1].hexbin(mw, c, gridsize=60, cmap="Greens", bins="log", mincnt=1)
axes[1].set_xlabel("MW (Da)"); axes[1].set_ylabel("Carbon Count"); axes[1].set_title("MW vs Carbon Count (log density)")
plt.colorbar(hb2, ax=axes[1], label="log10(count)")
fig.tight_layout(); fig.savefig(OUT/"06_property_space.png"); plt.close(fig)
print("Saved: 06_property_space.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 7: cho_class four-way property comparison
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Property Distributions by cho_class", fontsize=13, fontweight="bold")
for data, label, ax in [
    (mw, "MW (Da)", axes[0,0]), (c, "Carbon Count", axes[0,1]), (logp, "LogP", axes[0,2]),
    (tpsa, "TPSA (Å²)", axes[1,0]), (hba, "H-Bond Acceptors", axes[1,1]), (rot, "Rotatable Bonds", axes[1,2]),
]:
    for k in CHO_ORDER:
        if m[k].sum() > 50:
            ax.hist(data[m[k]], bins=50, color=CHO_COLORS[k], alpha=0.55, density=True,
                    label=f"{k} ({m[k].sum():,})", edgecolor="none")
    ax.set_xlabel(label); ax.set_ylabel("Density"); ax.set_title(label); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT/"07_cho_class_compare.png"); plt.close(fig)
print("Saved: 07_cho_class_compare.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 8: cho_class subset breakdown
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Kept-set Composition by cho_class", fontsize=13, fontweight="bold")
sizes = [m[k].sum() for k in CHO_ORDER]
axes[0].pie(sizes, labels=[f"{k}\n{s:,} ({s/total*100:.1f}%)" for k, s in zip(CHO_ORDER, sizes)],
            colors=[CHO_COLORS[k] for k in CHO_ORDER], startangle=90, wedgeprops=dict(width=0.5))
axes[0].set_title("cho_class")
# project scope: aromatic only
arom = is_arom.sum()
axes[1].pie([arom, total-arom],
            labels=[f"Aromatic (scope)\n{arom:,} ({arom/total*100:.1f}%)",
                    f"Aliphatic\n{total-arom:,} ({(total-arom)/total*100:.1f}%)"],
            colors=[C[0], C[7]], startangle=90, wedgeprops=dict(width=0.5))
axes[1].set_title("Aromatic scope (selected downstream)")
fig.tight_layout(); fig.savefig(OUT/"08_cho_breakdown.png"); plt.close(fig)
print("Saved: 08_cho_breakdown.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 9: v4 FILTER FUNNEL — what the changed conditions removed, + recovered classes
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("v5 Filter Impact — relaxed elements/charges, benzoin chemistry rules + vinyl_conj dropped",
             fontsize=13, fontweight="bold")

# (a) rejection-reason bar from the rejected set
rej = pd.read_csv(REJECT, usecols=["reject_reason"])
rc = rej.reject_reason.value_counts()
n_raw = total + len(rej)
ax = axes[0,0]
bars = ax.barh(range(len(rc)), rc.values, color=C[3], alpha=0.85)
ax.set_yticks(range(len(rc))); ax.set_yticklabels(rc.index, fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("molecules removed")
ax.set_title(f"Rejected by reason ({len(rej):,} of {n_raw:,} raw)")
for i, v in enumerate(rc.values):
    ax.text(v+300, i, f"{v:,}", va="center", fontsize=7.5)
fmt_x = mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}k")
ax.xaxis.set_major_formatter(fmt_x)

# (b) keep vs reject funnel
ax = axes[0,1]
ax.bar(["raw\npool", "v4\nkept"], [n_raw, total], color=[C[7], C[2]], alpha=0.85, width=0.6)
ax.set_ylabel("molecules"); ax.set_title(f"Funnel: {n_raw:,} → {total:,} ({total/n_raw*100:.0f}%)"); fmt(ax)
for i, v in enumerate([n_raw, total]):
    ax.text(i, v+3000, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

# (c) xtb_risk recovered classes (kept-but-tagged: nitro / azide / n_oxide)
ax = axes[1,0]
risk_tags = Counter()
for r in risk:
    for t in (r.split(",") if r else []):
        if t: risk_tags[t] += 1
n_tagged = int((risk != "").sum())
if risk_tags:
    items = risk_tags.most_common()
    ax.bar([k for k,_ in items], [v for _,v in items], color=C[1], alpha=0.85, width=0.5)
    for i,(k,v) in enumerate(items):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("count"); fmt(ax)
ax.set_title(f"xtb_risk recovered classes (kept+tagged: {n_tagged:,})")

# (d) newly-allowed elements recovered
ax = axes[1,1]
newel = {e: el_counter.get(e, 0) for e in ["B","Si","P","Se"]}
ax.bar(newel.keys(), newel.values(), color=C[5], alpha=0.85, width=0.5)
for i,(e,v) in enumerate(newel.items()):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("count"); ax.set_title(f"Newly-allowed elements recovered ({sum(newel.values()):,})"); fmt(ax)

fig.tight_layout(); fig.savefig(OUT/"09_v4_filter_impact.png"); plt.close(fig)
print("Saved: 09_v4_filter_impact.png")

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"v5 clean set        : {total:,}  (raw pool {n_raw:,}; kept {total/n_raw*100:.1f}%)")
print(f"MW median/mean      : {np.median(mw):.1f} / {np.mean(mw):.1f} Da  (cap ≤500)")
print(f"LogP median         : {np.median(logp):.2f}   TPSA median: {np.median(tpsa):.1f} Å²")
print(f"HBD/HBA/RotB median : {np.median(hbd):.0f} / {np.median(hba):.0f} / {np.median(rot):.0f}")
for k in CHO_ORDER:
    print(f"  {k:16s}: {m[k].sum():,} ({m[k].mean()*100:.1f}%)")
print(f"Aromatic scope      : {is_arom.sum():,} ({is_arom.mean()*100:.1f}%)")
print(f"xtb_risk tagged     : {n_tagged:,}  ({dict(risk_tags)})")
print(f"New elements B/Si/P/Se: {', '.join(f'{e}({el_counter.get(e,0):,})' for e in ['B','Si','P','Se'])}")
print(f"Top non-CHON els    : {', '.join(f'{e}({n:,})' for e,n in el_counter.most_common(8))}")
print(f"\nAll plots → {OUT}")
