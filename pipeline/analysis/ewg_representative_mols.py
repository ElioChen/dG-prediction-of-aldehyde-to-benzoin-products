#!/usr/bin/env python
"""Starting from scatter_xtb_vs_dft_EWGhighlight: pick representative highlighted (EWG)
molecules spanning the disagreement cloud, render each as a STANDALONE 2D structure
image (one per file), and emit a numbered version of the scatter that keys to them.

Representatives chosen to tell the story:
  - worst xTB over-stabilisation (xTB very exergonic, DFT endergonic) — top-left red cloud
  - the catastrophic single outlier (most negative DFT)
  - largest |gap|
  - a well-behaved EWG point near y=x (contrast: not every EWG fails)
  - mid-cloud fillers
"""
import os, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c/analysis"
SRC = f"{D}/merged_ewgtagged_20260619_022314.csv"
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

MOTIF = [("triflate", "triflate OS(=O)(=O)CF3"),
         ("sulfonyl-F", "sulfonyl_fluoride S(=O)(=O)F"),
         ("sulfonyl", "sulfonyl  S(=O)(=O)"),
         ("nitro", "nitro [N+](=O)[O-]")]

def motif_label(r):
    for name, col in MOTIF:
        if col in r and bool(r[col]):
            return name
    return "EWG"

def main():
    df = pd.read_csv(SRC)
    for c in ["dG_xtb_kcal", "dG_orca_kcal"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["dG_xtb_kcal", "dG_orca_kcal", "aldehyde_smiles"])
    ewg = df[df["is_ewg"] == True].copy()
    ewg["gap"] = ewg["dG_xtb_kcal"] - ewg["dG_orca_kcal"]   # neg = xTB over-exergonic
    ewg = ewg.reset_index(drop=True)

    picks = {}
    def add(row): picks[int(row.name)] = row
    # 1-2: worst over-stabilisation = most negative gap, among top DFT-endergonic cloud
    top = ewg[ewg.dG_orca_kcal > 25].sort_values("gap")
    for _, r in top.head(2).iterrows(): add(r)
    # 3: catastrophic outlier (most negative DFT)
    add(ewg.sort_values("dG_orca_kcal").iloc[0])
    # 4: largest |gap| overall
    add(ewg.iloc[ewg.gap.abs().idxmax()])
    # 5: well-behaved EWG near y=x (smallest |gap|, but real EWG)
    add(ewg.iloc[ewg.gap.abs().idxmin()])
    # 6: mid cloud (xTB ~ -20, DFT ~ +10)
    mid = ewg[(ewg.dG_xtb_kcal.between(-25, -12)) & (ewg.dG_orca_kcal.between(0, 18))]
    if len(mid): add(mid.sort_values("gap").iloc[len(mid)//2])
    reps = list(picks.values())[:6]
    reps = pd.DataFrame(reps).reset_index(drop=True)
    reps["motif"] = reps.apply(motif_label, axis=1)
    reps["num"] = range(1, len(reps) + 1)

    # ---- numbered scatter (standalone) ----
    plt.figure(figsize=(8.2, 8.0))
    keep = df[df["is_ewg"] != True]
    plt.scatter(keep.dG_xtb_kcal, keep.dG_orca_kcal, s=10, alpha=0.35,
                color="#4d77b0", edgecolors="none", label=f"kept (n={len(keep)})")
    plt.scatter(ewg.dG_xtb_kcal, ewg.dG_orca_kcal, s=14, alpha=0.6,
                color="#d7301f", edgecolors="none", label=f"EWG (n={len(ewg)})")
    lim = [-105, 85]
    plt.plot(lim, lim, "--", color="0.4", lw=1, label="y = x")
    for _, r in reps.iterrows():
        plt.scatter([r.dG_xtb_kcal], [r.dG_orca_kcal], s=130, facecolors="none",
                    edgecolors="black", linewidths=1.8, zorder=5)
        plt.annotate(str(r.num), (r.dG_xtb_kcal, r.dG_orca_kcal),
                     textcoords="offset points", xytext=(7, 5), fontsize=12,
                     fontweight="bold", zorder=6)
    plt.xlim(lim); plt.ylim(lim); plt.gca().set_aspect("equal")
    plt.xlabel("ΔG xTB (GFN2)  [kcal/mol]"); plt.ylabel("ΔG DFT (r2SCAN-3c//xTB)  [kcal/mol]")
    plt.title("xTB vs DFT — EWG highlighted, with representative molecules numbered")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    fS = f"{D}/scatter_EWG_numbered_reps_{TS}.png"; plt.savefig(fS, dpi=150); plt.close()

    # ---- one standalone structure image per representative ----
    files = []
    for _, r in reps.iterrows():
        m = Chem.MolFromSmiles(r.aldehyde_smiles)
        if m is None:
            continue
        AllChem.Compute2DCoords(m)
        dr = rdMolDraw2D.MolDraw2DCairo(560, 460)
        opt = dr.drawOptions(); opt.legendFontSize = 18
        name = (str(r.get("aldehyde_name", "")) or "")[:46]
        legend = (f"#{r.num}  {r.motif}   "
                  f"xTB {r.dG_xtb_kcal:+.1f} | DFT {r.dG_orca_kcal:+.1f} | "
                  f"gap {r.gap:+.1f} kcal/mol")
        rdMolDraw2D.PrepareAndDrawMolecule(dr, m, legend=legend)
        dr.FinishDrawing()
        f = f"{D}/repmol_{r.num}_{r.motif}_{TS}.png"
        open(f, "wb").write(dr.GetDrawingText()); files.append(f)

    reps_out = reps[["num", "motif", "aldehyde_name", "aldehyde_smiles",
                     "dG_xtb_kcal", "dG_orca_kcal", "gap"]]
    reps_out.to_csv(f"{D}/representative_ewg_mols_{TS}.csv", index=False)
    print(reps_out.to_string())
    print("\nscatter:", fS)
    print("mol images:", *files, sep="\n  ")

if __name__ == "__main__":
    main()
