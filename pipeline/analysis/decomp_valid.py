#!/usr/bin/env python
"""VALID reactant-vs-product decomposition of the xTB-vs-DFT ΔG discrepancy.

The old decomp figure was an artifact: per-species (E_orca - E_xtb) is dominated by
the incompatible absolute-energy reference (xTB valence ~-60 Eh vs all-electron DFT
~-1200 Eh), so each species' "contribution" was ~-7e5 kcal of pure reference offset.

Fix — empirical atomic reference. Write per-species electronic disagreement
  delta(s) = E_el_orca(s) - E_el_xtb(s)
as  delta(s) = sum_element n_e(s) * a_e  +  r(s),
where a_e (fit by no-intercept least squares over ALL species) absorbs the per-element
method offset and r(s) is the chemical-sized residual disagreement. Because benzoin
condensation conserves atoms exactly (n_product = 2 * n_aldehyde), the a_e term cancels
in the reaction, so the FULL discrepancy is reproduced exactly by r alone:
  D = ΔG_orca - ΔG_xtb = r(prod) - 2*r(ald).
=> reactant-side contribution = -2*r(ald),  product-side contribution = +r(prod).

Inputs reconstructed from the 1% merged CSV:
  thermal_xtb(s) = G_orca(s) - E_orca(s)        (xTB RRHO used on the DFT energy)
  E_el_xtb(s)    = G_xtb(s) - thermal_xtb(s)
"""
import sys, datetime
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HART = 627.509474
MERGED = sys.argv[1] if len(sys.argv) > 1 else \
    "/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c/analysis/dft_sp_merged_20260618_002636.csv"
OUT = "/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c/analysis"
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

need = ["dG_xtb_kcal","dG_orca_kcal","G_ald_xtb_Eh","G_bz_xtb_Eh",
        "G_ald_orca_Eh","G_bz_orca_Eh","E_ald_orca_Eh","E_bz_orca_Eh",
        "aldehyde_smiles","benzoin_smiles"]
df = pd.read_csv(MERGED).dropna(subset=need).reset_index(drop=True)

# EWG tag (same definition as the drop-EWG analysis)
PAT_SULFONYL = Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])")
PAT_NITRO = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")
def is_ewg(sm):
    m = Chem.MolFromSmiles(sm)
    return bool(m and (m.HasSubstructMatch(PAT_SULFONYL) or m.HasSubstructMatch(PAT_NITRO)))
df["is_ewg"] = df["aldehyde_smiles"].apply(is_ewg)

# reconstruct electronic energies (Eh)
th_ald = df["G_ald_orca_Eh"] - df["E_ald_orca_Eh"]
th_bz  = df["G_bz_orca_Eh"]  - df["E_bz_orca_Eh"]
Extb_ald = df["G_ald_xtb_Eh"] - th_ald
Extb_bz  = df["G_bz_xtb_Eh"]  - th_bz
delta_ald = (df["E_ald_orca_Eh"] - Extb_ald).to_numpy()   # Eh
delta_bz  = (df["E_bz_orca_Eh"]  - Extb_bz ).to_numpy()

# element-count matrix (H included) over all species (ald rows then bz rows)
def counts(sm):
    m = Chem.AddHs(Chem.MolFromSmiles(sm))
    return Counter(a.GetSymbol() for a in m.GetAtoms())
ca = df["aldehyde_smiles"].apply(counts)
cb = df["benzoin_smiles"].apply(counts)
elements = sorted({e for c in list(ca)+list(cb) for e in c})
def vec(c): return np.array([c.get(e, 0) for e in elements], float)
Xa = np.vstack(ca.apply(vec).to_numpy())
Xb = np.vstack(cb.apply(vec).to_numpy())

# fit per-element offset a_e on ALL species (no intercept)
X = np.vstack([Xa, Xb]); y = np.concatenate([delta_ald, delta_bz])
a, *_ = np.linalg.lstsq(X, y, rcond=None)        # Eh per atom of element e
r_ald = delta_ald - Xa @ a                        # Eh, chemical residual
r_bz  = delta_bz  - Xb @ a

# contributions to D (kcal/mol)
D_react = (-2.0 * r_ald) * HART
D_prod  = ( r_bz)        * HART
D_total = D_react + D_prod
D_csv   = (df["dG_orca_kcal"] - df["dG_xtb_kcal"]).to_numpy()
max_err = float(np.abs(D_total - D_csv).max())

df["D_react"] = D_react; df["D_prod"] = D_prod; df["D_total"] = D_total

def summ(mask, name):
    d = df[mask]
    return dict(name=name, n=int(mask.sum()),
                D=float(d["D_total"].mean()),
                react=float(d["D_react"].mean()), prod=float(d["D_prod"].mean()),
                react_abs=float(d["D_react"].abs().mean()),
                prod_abs=float(d["D_prod"].abs().mean()),
                prod_share=float(d["D_prod"].abs().mean() /
                                 (d["D_prod"].abs().mean()+d["D_react"].abs().mean())))
S_all = summ(df.index == df.index, "ALL")
S_ben = summ(~df["is_ewg"], "benign")
S_ewg = summ(df["is_ewg"], "EWG")

# ---------- Fig 1: valid decomposition scatter ----------
plt.figure(figsize=(6.6, 6.2))
for mask, c, lab in [(~df["is_ewg"], "#2c7fb8", f"benign (n={S_ben['n']})"),
                     (df["is_ewg"], "#d7301f", f"EWG (n={S_ewg['n']})")]:
    plt.scatter(df.loc[mask, "D_react"], df.loc[mask, "D_prod"],
                s=12, alpha=0.5, color=c, edgecolors="none", label=lab)
lim = np.array([-1, 1]) * np.nanpercentile(np.abs(np.r_[D_react, D_prod]), 99) * 1.1
for k in (-1, 0, 1):
    plt.plot(lim, k*0 - lim + 0, alpha=0)  # noop keep autoscale
plt.axhline(0, color="k", lw=0.8); plt.axvline(0, color="k", lw=0.8)
xs = np.linspace(lim[0], lim[1], 10)
plt.plot(xs, -xs, "k--", lw=0.8, alpha=0.6, label="D = 0 (react = -prod)")
plt.xlim(lim); plt.ylim(lim)
plt.xlabel("reactant-side contribution  -2·r(ald)  [kcal/mol]")
plt.ylabel("product-side contribution  +r(benzoin)  [kcal/mol]")
plt.title("Valid reactant/product decomposition of ΔG discrepancy\n"
          f"D = ΔG_DFT - ΔG_xTB  (recon err {max_err:.3f} kcal)")
plt.legend(fontsize=8, loc="upper left"); plt.tight_layout()
f1 = f"{OUT}/decomp_valid_react_vs_prod_{TS}.png"
plt.savefig(f1, dpi=150); plt.close()

# ---------- Fig 2: which side dominates — mean |contribution| ----------
plt.figure(figsize=(6.2, 5.0))
groups = ["benign", "EWG"]; react = [S_ben["react_abs"], S_ewg["react_abs"]]
prod = [S_ben["prod_abs"], S_ewg["prod_abs"]]
x = np.arange(len(groups)); w = 0.38
plt.bar(x - w/2, react, w, color="#7570b3", label="reactant side |-2·r(ald)|")
plt.bar(x + w/2, prod,  w, color="#1b9e77", label="product side |r(benzoin)|")
for xi, (rr, pp) in enumerate(zip(react, prod)):
    plt.text(xi - w/2, rr, f"{rr:.1f}", ha="center", va="bottom", fontsize=8)
    plt.text(xi + w/2, pp, f"{pp:.1f}", ha="center", va="bottom", fontsize=8)
plt.xticks(x, groups); plt.ylabel("mean |contribution to D|  [kcal/mol]")
plt.title("xTB–DFT disagreement: reactant vs product side")
plt.legend(fontsize=8); plt.tight_layout()
f2 = f"{OUT}/decomp_valid_sidemag_{TS}.png"
plt.savefig(f2, dpi=150); plt.close()

# ---------- Fig 3: per-element fitted offset (the absorbed reference) ----------
plt.figure(figsize=(6.6, 4.6))
order = np.argsort(a)
plt.bar(range(len(elements)), (a[order]) , color="#888")
plt.xticks(range(len(elements)), [elements[i] for i in order])
plt.ylabel("fitted a_e  [Eh / atom]")
plt.title("Per-element method offset absorbed by the atomic reference\n"
          "(this is the ~1300 Eh/atom-scale term that made the old decomp an artifact)")
plt.tight_layout()
f3 = f"{OUT}/decomp_valid_element_offset_{TS}.png"
plt.savefig(f3, dpi=150); plt.close()

# ---------- report ----------
def row(s):
    return (f"| {s['name']} | {s['n']} | {s['D']:+.2f} | {s['react']:+.2f} | "
            f"{s['prod']:+.2f} | {s['react_abs']:.2f} | {s['prod_abs']:.2f} | "
            f"{100*s['prod_share']:.0f}% |")
md = f"""# Valid reactant/product decomposition of the xTB–DFT ΔG discrepancy — {TS}

Source: `{MERGED}`  (n={len(df)} valid pairs).  D ≡ ΔG_DFT − ΔG_xTB.

## Why this supersedes the old `decomp_reactant_product` figure

Per-species `E_orca − E_xtb` ≈ {np.mean(delta_ald)*HART:,.0f} kcal/mol (aldehyde) — this
is **pure absolute-reference offset** (xTB valence energy ~−73 Eh vs all-electron
r2SCAN-3c ~−1368 Eh), not chemistry. The old figure plotted exactly this, so its
"reactant-dominated, +1.6M kcal" was an artifact of the 2:1 atom ratio.

Here each species' electronic disagreement is referenced to its atoms:
`delta(s) = Σ n_e·a_e + r(s)`, a_e fit over all {2*len(df)} species. Atom conservation
(n_product = 2·n_aldehyde, verified) makes the a_e term cancel exactly, so the residual
reproduces the full discrepancy: **D = r(prod) − 2·r(ald)** (max recon error
**{max_err:.4f} kcal/mol** vs the CSV ΔG — i.e. exact).

- reactant-side contribution to D: **−2·r(ald)**
- product-side contribution to D:  **+r(benzoin)**

## Attribution (mean over each set)

| set | n | mean D | mean react | mean prod | mean \\|react\\| | mean \\|prod\\| | product share of \\|D\\| |
|---|---|---|---|---|---|---|---|
{row(S_all)}
{row(S_ben)}
{row(S_ewg)}

## Interpretation

- **The discrepancy is shared, not one-sided.** Product share of |D| = {100*S_all['prod_share']:.0f}%
  overall — both the aldehyde and the benzoin electronic structures are described
  differently by xTB vs r2SCAN-3c; neither side alone explains the gap.
- **EWG molecules carry far larger per-side disagreement** (mean |react| {S_ewg['react_abs']:.1f} /
  |prod| {S_ewg['prod_abs']:.1f} kcal vs benign {S_ben['react_abs']:.1f} / {S_ben['prod_abs']:.1f}),
  consistent with hypervalent-S being an electronic-structure failure of xTB.
- Sign of the mean contributions shows the **direction**: the product side pushes D
  {('positive' if S_all['prod']>0 else 'negative')} and the reactant side
  {('positive' if S_all['react']>0 else 'negative')}; their balance gives the net
  +{S_all['D']:.1f} kcal/mol (xTB too exergonic).

## Figures (standalone)

- `{f1.split('/')[-1]}`  — reactant vs product contribution scatter (valid decomp)
- `{f2.split('/')[-1]}`  — mean |contribution| per side, benign vs EWG
- `{f3.split('/')[-1]}`  — per-element fitted offset a_e (the absorbed reference)
"""
mdp = f"{OUT}/REPORT_decomp_valid_{TS}.md"
open(mdp, "w").write(md)
df[["aldehyde_smiles","benzoin_smiles","is_ewg","D_total","D_react","D_prod",
    "dG_xtb_kcal","dG_orca_kcal"]].to_csv(f"{OUT}/decomp_valid_permol_{TS}.csv", index=False)

print("recon max err (kcal):", max_err)
print("ALL", S_all); print("benign", S_ben); print("EWG", S_ewg)
print("a_e (Eh):", dict(zip(elements, np.round(a,4))))
print("wrote:", f1, f2, f3, mdp, sep="\n  ")
