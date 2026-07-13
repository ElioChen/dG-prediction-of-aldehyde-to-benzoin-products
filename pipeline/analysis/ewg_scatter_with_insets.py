#!/usr/bin/env python
"""xTB-vs-DFT EWG scatter with 8-10 representative molecule structures as INSETS,
each connected by an arrow to its data point. Smaller thumbnails; only chemically
VALID benzoin products (excludes the wrong-carbonyl coupling bug rows)."""
import os, io, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c/analysis"
SRC = f"{D}/merged_ewgtagged_20260619_022314.csv"
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MOTIF = [("triflate", "triflate OS(=O)(=O)CF3"), ("sulfonyl-F", "sulfonyl_fluoride S(=O)(=O)F"),
         ("sulfonyl", "sulfonyl  S(=O)(=O)"), ("nitro", "nitro [N+](=O)[O-]")]
CARB = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
LIM = [-110, 90]

def motif_label(r):
    for n, c in MOTIF:
        if c in r and bool(r[c]): return n
    return "EWG"

def valid_product(smi):
    b = Chem.MolFromSmiles(str(smi))
    return b is not None and len(b.GetSubstructMatches(CARB)) == 0  # no leftover free R-CHO

def mol_png(smi, legend):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    AllChem.Compute2DCoords(m)
    dr = rdMolDraw2D.MolDraw2DCairo(360, 300)
    dr.drawOptions().legendFontSize = 26
    rdMolDraw2D.PrepareAndDrawMolecule(dr, m, legend=legend)
    dr.FinishDrawing()
    return mpimg.imread(io.BytesIO(dr.GetDrawingText()), format="png")

def to_frac(x, y):
    return ((x - LIM[0]) / (LIM[1] - LIM[0]), (y - LIM[0]) / (LIM[1] - LIM[0]))

def main():
    df = pd.read_csv(SRC)
    for c in ["dG_xtb_kcal", "dG_orca_kcal"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["dG_xtb_kcal", "dG_orca_kcal", "aldehyde_smiles"])
    ewg = df[df["is_ewg"] == True].copy()
    ewg = ewg[ewg["benzoin_smiles"].apply(valid_product)].copy()   # exclude wrong-coupling rows
    ewg["gap"] = ewg.dG_xtb_kcal - ewg.dG_orca_kcal
    ewg = ewg.reset_index(drop=True)

    picks = {}
    def add(r):
        if r is not None and int(r.name) not in picks: picks[int(r.name)] = r
    # 3 extreme over-stabilisers across the top DFT-endergonic cloud (spread in xTB)
    top = ewg[ewg.dG_orca_kcal > 38].sort_values("gap")
    if len(top):
        add(top.iloc[0])
        add(top.sort_values("dG_xtb_kcal").iloc[-1])     # rightmost of the top cloud
        add(top.iloc[len(top)//2])
    add(ewg.sort_values("dG_orca_kcal").iloc[0])         # catastrophic outlier
    add(ewg.iloc[ewg.gap.abs().idxmax()])                # largest |gap|
    # 2 well-behaved EWG near y=x
    nd = ewg.iloc[ewg.gap.abs().sort_values().index]
    add(nd.iloc[0]); add(nd[nd.dG_orca_kcal > 5].iloc[0] if (nd.dG_orca_kcal > 5).any() else nd.iloc[1])
    # mid cloud fillers
    mid = ewg[(ewg.dG_xtb_kcal.between(-28, -10)) & (ewg.dG_orca_kcal.between(-2, 22))].sort_values("gap")
    for q in (0.25, 0.6, 0.85):
        if len(mid): add(mid.iloc[int(len(mid)*q)])
    reps = pd.DataFrame(list(picks.values())[:10]).reset_index(drop=True)
    reps["motif"] = reps.apply(motif_label, axis=1)
    reps = reps.sort_values("dG_xtb_kcal").reset_index(drop=True)
    reps["num"] = range(1, len(reps) + 1)

    # perimeter slots (axes fraction) in empty regions; assign each rep to nearest slot
    slots = [(0.12, 0.88), (0.35, 0.90), (0.58, 0.90), (0.90, 0.72), (0.92, 0.50),
             (0.91, 0.26), (0.38, 0.06), (0.62, 0.06), (0.10, 0.44), (0.10, 0.22)]
    used = set(); assign = {}
    for _, r in reps.iterrows():
        fx, fy = to_frac(r.dG_xtb_kcal, r.dG_orca_kcal)
        best = min((s for s in slots if s not in used),
                   key=lambda s: (s[0]-fx)**2 + (s[1]-fy)**2)
        used.add(best); assign[int(r.num)] = best

    fig, ax = plt.subplots(figsize=(15.5, 13.5))
    keep = df[df["is_ewg"] != True]
    ax.scatter(keep.dG_xtb_kcal, keep.dG_orca_kcal, s=9, alpha=0.28, color="#4d77b0",
               edgecolors="none", label=f"kept (n={len(keep)})")
    ax.scatter(ewg.dG_xtb_kcal, ewg.dG_orca_kcal, s=15, alpha=0.5, color="#d7301f",
               edgecolors="none", label=f"EWG (n={len(ewg)})")
    ax.plot(LIM, LIM, "--", color="0.4", lw=1, label="y = x")
    ax.set_xlim(LIM); ax.set_ylim(LIM); ax.set_aspect("equal")
    ax.set_xlabel("ΔG xTB (GFN2)  [kcal/mol]", fontsize=13)
    ax.set_ylabel("ΔG DFT (r2SCAN-3c//xTB)  [kcal/mol]", fontsize=13)
    ax.set_title(f"xTB vs DFT — EWG highlighted, {len(reps)} representative structures inset "
                 "(valid products only)", fontsize=15, pad=28)
    ax.legend(loc="lower right", fontsize=12)

    for _, r in reps.iterrows():
        ax.scatter([r.dG_xtb_kcal], [r.dG_orca_kcal], s=140, facecolors="none",
                   edgecolors="black", linewidths=1.8, zorder=6)
        img = mol_png(r.aldehyde_smiles,
                      f"#{r.num} {r.motif}: xTB {r.dG_xtb_kcal:+.0f}/DFT {r.dG_orca_kcal:+.0f}")
        if img is None: continue
        ab = AnnotationBbox(OffsetImage(img, zoom=0.40), (r.dG_xtb_kcal, r.dG_orca_kcal),
                            xybox=assign[int(r.num)], xycoords="data", boxcoords="axes fraction",
                            box_alignment=(0.5, 0.5), pad=0.2,
                            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2,
                                            connectionstyle="arc3,rad=0.12"), zorder=7,
                            bboxprops=dict(edgecolor="0.5", lw=0.7))
        ax.add_artist(ab)
    plt.tight_layout()
    f = f"{D}/scatter_EWG_with_mol_insets_{TS}.png"; plt.savefig(f, dpi=150); plt.close()
    print("wrote:", f, "  n_reps=", len(reps))
    print(reps[["num", "motif", "dG_xtb_kcal", "dG_orca_kcal", "gap"]].to_string())

if __name__ == "__main__":
    main()
