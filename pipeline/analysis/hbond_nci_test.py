#!/usr/bin/env python
"""Test whether the product-side xTB-vs-DFT disagreement r(benzoin) is associated with
the intramolecular alpha-hydroxyketone O-H...O=C hydrogen bond (an NCI that xTB and DFT
may describe differently AT THE SAME geometry).

r(benzoin) here = product-side electronic disagreement at the FIXED xTB geometry
(thermal removed), from the same atom-referenced decomposition as decomp_valid.py.
H-bond geometry is perceived purely from the saved xTB-optimised benzoin xyz (no atom-
ordering assumptions): classify O atoms by neighbours, then for each hydroxyl O-H find
the nearest carbonyl-type acceptor O and measure H...O distance and O-H...O angle.
"""
import os, datetime
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HART = 627.509474
RES = "/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c"
OUT = f"{RES}/analysis"
MERGED = f"{OUT}/dft_sp_merged_20260618_002636.csv"
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
XYZ = f"{RES}/benzoin_xyz"

# covalent radii (Angstrom) for simple bond perception
RCOV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07,
            S=1.05, Cl=1.02, Se=1.20, Br=1.20, I=1.39)

def read_xyz(path):
    L = open(path).read().splitlines()
    n = int(L[0].split()[0])
    sym = []; xyz = []
    for l in L[2:2+n]:
        p = l.split(); sym.append(p[0]); xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return sym, np.array(xyz)

def neighbors(sym, X):
    n = len(sym); D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    nb = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            cut = (RCOV.get(sym[i], 0.77) + RCOV.get(sym[j], 0.77)) * 1.3
            if 0.4 < D[i, j] < cut:
                nb[i].append(j); nb[j].append(i)
    return nb, D

def hbond_geom(path):
    """Return dict with the shortest intramolecular hydroxyl-O-H...O=C contact."""
    sym, X = read_xyz(path); nb, D = neighbors(sym, X)
    hydroxyl = []  # (O_idx, H_idx)
    carbonyl_O = []  # O_idx (=O on a C that has another heavy neighbour)
    for i, s in enumerate(sym):
        if s != "O":
            continue
        hs = [j for j in nb[i] if sym[j] == "H"]
        cs = [j for j in nb[i] if sym[j] == "C"]
        if len(hs) == 1 and len(cs) == 1:                      # hydroxyl O-H
            hydroxyl.append((i, hs[0]))
        elif len(nb[i]) == 1 and sym[nb[i][0]] == "C" and D[i, nb[i][0]] < 1.30:  # carbonyl =O
            carbonyl_O.append(i)
    best = None
    for (oh, h) in hydroxyl:
        for oc in carbonyl_O:
            dHO = D[h, oc]
            v1 = X[oh] - X[h]; v2 = X[oc] - X[h]
            ang = np.degrees(np.arccos(np.clip(
                v1 @ v2 / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9), -1, 1)))
            dOO = D[oh, oc]
            if best is None or dHO < best["dHO"]:
                best = dict(dHO=float(dHO), angle=float(ang), dOO=float(dOO))
    if best is None:
        return dict(dHO=np.nan, angle=np.nan, dOO=np.nan,
                    n_hydroxyl=len(hydroxyl), n_carbonyl=len(carbonyl_O))
    best.update(n_hydroxyl=len(hydroxyl), n_carbonyl=len(carbonyl_O))
    return best

# ---- recompute atom-referenced product-side residual r(benzoin) ----
need = ["G_ald_xtb_Eh","G_bz_xtb_Eh","G_ald_orca_Eh","G_bz_orca_Eh",
        "E_ald_orca_Eh","E_bz_orca_Eh","aldehyde_smiles","benzoin_smiles",
        "benzoin_xyz_file","dG_xtb_kcal","dG_orca_kcal"]
df = pd.read_csv(MERGED).dropna(subset=need).reset_index(drop=True)
PAT_S = Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])")
PAT_N = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")
df["is_ewg"] = df["aldehyde_smiles"].apply(
    lambda s: bool((m:=Chem.MolFromSmiles(s)) and (m.HasSubstructMatch(PAT_S) or m.HasSubstructMatch(PAT_N))))

th_ald = df["G_ald_orca_Eh"]-df["E_ald_orca_Eh"]; th_bz = df["G_bz_orca_Eh"]-df["E_bz_orca_Eh"]
delta_ald = (df["E_ald_orca_Eh"]-(df["G_ald_xtb_Eh"]-th_ald)).to_numpy()
delta_bz  = (df["E_bz_orca_Eh"] -(df["G_bz_xtb_Eh"] -th_bz )).to_numpy()
def cvec(sm, elements):
    c = Counter(a.GetSymbol() for a in Chem.AddHs(Chem.MolFromSmiles(sm)).GetAtoms())
    return np.array([c.get(e, 0) for e in elements], float)
els = sorted({a.GetSymbol() for sm in pd.concat([df.aldehyde_smiles, df.benzoin_smiles])
              for a in Chem.AddHs(Chem.MolFromSmiles(sm)).GetAtoms()})
Xa = np.vstack([cvec(s, els) for s in df.aldehyde_smiles])
Xb = np.vstack([cvec(s, els) for s in df.benzoin_smiles])
a, *_ = np.linalg.lstsq(np.vstack([Xa, Xb]), np.concatenate([delta_ald, delta_bz]), rcond=None)
df["r_bz_kcal"] = (delta_bz - Xb @ a) * HART       # product-side residual (signed)
df["D_prod_kcal"] = df["r_bz_kcal"]                 # contribution to D = +r(benzoin)

# ---- perceive H-bond on each xTB benzoin geometry ----
rows = []
for _, rr in df.iterrows():
    p = f"{XYZ}/{os.path.basename(rr['benzoin_xyz_file'])}"
    rows.append(hbond_geom(p) if os.path.exists(p) else
                dict(dHO=np.nan, angle=np.nan, dOO=np.nan, n_hydroxyl=np.nan, n_carbonyl=np.nan))
hb = pd.DataFrame(rows); df = pd.concat([df.reset_index(drop=True), hb], axis=1)
df["has_hbond"] = (df["dHO"] < 2.2) & (df["angle"] > 120)

ok = df.dropna(subset=["dHO"])
print(f"parsed geoms: {len(df)}  with detectable hydroxyl+carbonyl: {len(ok)}  "
      f"H-bond formed (dHO<2.2,ang>120): {int(df['has_hbond'].sum())} "
      f"({100*df['has_hbond'].mean():.0f}%)")

def corr(d, col):
    dd = d.dropna(subset=[col, "D_prod_kcal"])
    if len(dd) < 5: return (np.nan, np.nan, len(dd))
    r, p = stats.pearsonr(dd[col], dd["D_prod_kcal"]); return (r, p, len(dd))

report = {}
for name, mask in [("ALL", df.index==df.index), ("benign", ~df.is_ewg), ("EWG", df.is_ewg)]:
    d = df[mask]
    report[name] = dict(
        n=int(mask.sum()),
        pct_hbond=100*float(d["has_hbond"].mean()),
        dHO_mean=float(d["dHO"].mean(skipna=True)),
        Dprod_hb=float(d[d["has_hbond"]]["D_prod_kcal"].mean()) if d["has_hbond"].any() else np.nan,
        Dprod_nohb=float(d[~d["has_hbond"]]["D_prod_kcal"].mean()) if (~d["has_hbond"]).any() else np.nan,
        r_dHO=corr(d, "dHO"), r_ang=corr(d, "angle"))

# ---- Fig 1: D_prod vs H-bond H...O distance ----
plt.figure(figsize=(6.6, 5.2))
for mask, c, lab in [(~df.is_ewg, "#2c7fb8", "benign"), (df.is_ewg, "#d7301f", "EWG")]:
    d = df[mask].dropna(subset=["dHO"])
    plt.scatter(d["dHO"], d["D_prod_kcal"], s=12, alpha=0.5, color=c, edgecolors="none", label=lab)
plt.axhline(0, color="k", lw=0.8)
plt.axvline(2.2, color="gray", ls=":", lw=1, label="H-bond cutoff 2.2 Å")
plt.xlabel("intramolecular hydroxyl  H···O=C  distance  [Å]  (xTB geometry)")
plt.ylabel("product-side disagreement  r(benzoin)  [kcal/mol]")
plt.title("Is product-side xTB–DFT error linked to the α-hydroxyketone H-bond?")
plt.legend(fontsize=8); plt.tight_layout()
f1 = f"{OUT}/hbond_Dprod_vs_dHO_{TS}.png"; plt.savefig(f1, dpi=150); plt.close()

# ---- Fig 2: distribution of H...O distance, benign vs EWG ----
plt.figure(figsize=(6.4, 4.8))
for mask, c, lab in [(~df.is_ewg, "#2c7fb8", "benign"), (df.is_ewg, "#d7301f", "EWG")]:
    d = df[mask].dropna(subset=["dHO"])
    plt.hist(d["dHO"], bins=40, range=(1.4, 4.5), alpha=0.55, color=c, density=True, label=lab)
plt.axvline(2.2, color="gray", ls=":", lw=1)
plt.xlabel("hydroxyl H···O=C distance [Å] (xTB geometry)")
plt.ylabel("density"); plt.legend(fontsize=8)
plt.title("α-hydroxyketone H-bond geometry: benign vs EWG")
plt.tight_layout(); f2 = f"{OUT}/hbond_dHO_hist_{TS}.png"; plt.savefig(f2, dpi=150); plt.close()

md = [f"# Intramolecular H-bond / NCI vs product-side disagreement — {TS}", "",
      f"Source: `{MERGED}`, xTB benzoin geometries in `benzoin_xyz/`.",
      "r(benzoin) = product-side electronic disagreement at FIXED xTB geometry "
      "(thermal removed; atom-referenced).", "",
      f"- benzoin geometries parsed: **{len(df)}**, with detectable hydroxyl+carbonyl "
      f"**{len(ok)}**, intramolecular H-bond formed (H···O<2.2 Å, ∠>120°): "
      f"**{int(df['has_hbond'].sum())} ({100*df['has_hbond'].mean():.0f}%)**", "",
      "| set | n | % H-bond | mean H···O (Å) | mean r(bz) H-bonded | mean r(bz) no-Hbond | Pearson r(bz) vs H···O |",
      "|---|---|---|---|---|---|---|"]
for k, s in report.items():
    md.append(f"| {k} | {s['n']} | {s['pct_hbond']:.0f}% | {s['dHO_mean']:.2f} | "
              f"{s['Dprod_hb']:+.1f} | {s['Dprod_nohb']:+.1f} | "
              f"{s['r_dHO'][0]:+.3f} (p={s['r_dHO'][1]:.1e}, n={s['r_dHO'][2]}) |")
md += ["", "## Figures", f"- `{f1.split('/')[-1]}`  — r(benzoin) vs H···O distance",
       f"- `{f2.split('/')[-1]}`  — H···O distance distribution, benign vs EWG"]
open(f"{OUT}/REPORT_hbond_nci_{TS}.md", "w").write("\n".join(md))
df.to_csv(f"{OUT}/hbond_nci_permol_{TS}.csv", index=False)
print("\n".join(md[5:]))
print("wrote:", f1, f2, f"{OUT}/REPORT_hbond_nci_{TS}.md", sep="\n  ")
