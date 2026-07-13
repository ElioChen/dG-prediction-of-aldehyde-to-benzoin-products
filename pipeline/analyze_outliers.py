#!/usr/bin/env python
"""
Characterise the NON-EWG outliers of the xTB-vs-DFT benzoin ΔG validation.

The EWG (hypervalent-S / nitro) motifs are the known dominant xTB failure
(see REPORT_corr_dropEWG). This tool asks the next question: *after removing
EWG, what still drives the large xTB-DFT residuals?* It

  1. takes a per-molecule table with dG_xtb_kcal, dG_orca_kcal and (ideally) the
     reactant/product decomposition D_react / D_prod from decomp_valid.py;
  2. flags EWG with the same SMARTS, isolates the benign (non-EWG) subset;
  3. defines outliers robustly (modified z on the residual, |z|>Z, default 3.5);
  4. computes a battery of structural descriptors on aldehyde + benzoin;
  5. reports which descriptors are ENRICHED among the non-EWG outliers vs the
     non-EWG background (enrichment ratio + counts), and attributes each outlier
     to the reactant- or product-side using D_react / D_prod;
  6. writes a markdown report, an outlier per-molecule CSV, and standalone figures
     (one chart per file).

Usage:
    python analyze_outliers.py INPUT.csv [--outdir DIR] [--z 3.5]
INPUT may be a decomp_valid_permol CSV (preferred: has D_react/D_prod) or any
merged CSV with aldehyde_smiles, benzoin_smiles, dG_xtb_kcal, dG_orca_kcal.
For the full SP set: run decomp_valid.py first, then point this at its permol CSV.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# ── EWG definition (identical to decomp_valid.py / REPORT_corr_dropEWG) ──────────
PAT_SULFONYL = Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])")
PAT_NITRO = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")


def is_ewg(sm: str) -> bool:
    m = Chem.MolFromSmiles(sm or "")
    if m is None:
        return False
    return m.HasSubstructMatch(PAT_SULFONYL) or m.HasSubstructMatch(PAT_NITRO)


# ── Structural descriptors (the candidate non-EWG error drivers) ─────────────────
_SMARTS = {
    "aromatic_O(furan)": "[o]",
    "aromatic_S(thiophene)": "[s]",
    "aromatic_N(azine/azole)": "[n]",
    "nitrile_C#N": "[CX2]#[NX1]",
    "ester/acyl_O": "[CX3](=[OX1])[OX2H0]",
    "N/O-formyl(bugclass)": "[CX3H1](=O)[#7,#8]",
    "thioether_S": "[#16X2]",
    "halothio/SF": "[#16X2]F",
    "azo/diazo_N=N": "[NX2]=[NX2]",
    "imine_C=N": "[CX3]=[NX2]",
}
_SMARTS = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}
_HBD = Chem.MolFromSmarts("[OX2H,NX3;H1,H2]")
_OH = Chem.MolFromSmarts("[OX2H]")


def descriptors(ald_smi: str, bz_smi: str) -> dict:
    """Structural features; substituent classes from aldehyde, size/flex/HBD from product."""
    d: dict = {}
    ald = Chem.MolFromSmiles(ald_smi or "")
    bz = Chem.MolFromSmiles(bz_smi or "")
    if ald is None:
        d["ald_parse_fail"] = True
        return d
    # substituent-class flags on the reactant (defines the chemistry)
    for name, pat in _SMARTS.items():
        d[name] = ald.HasSubstructMatch(pat)
    syms = [a.GetSymbol() for a in ald.GetAtoms()]
    for el in ("F", "Cl", "Br", "I", "B", "Si", "P", "Se"):
        d[f"has_{el}"] = el in syms
    d["heavy_halogen(Br/I)"] = ("Br" in syms) or ("I" in syms)
    d["formal_charge!=0"] = Chem.GetFormalCharge(ald) != 0
    ri = ald.GetRingInfo()
    d["polycyclic(>=2 rings)"] = ri.NumRings() >= 2
    # size / flexibility / H-bonding on the product benzoin (what is optimised)
    if bz is not None:
        d["bz_heavy"] = bz.GetNumHeavyAtoms()
        d["bz_nrot"] = rdMolDescriptors.CalcNumRotatableBonds(bz)
        d["bz_HBD"] = len(bz.GetSubstructMatches(_HBD))
        d["bz_nOH"] = len(bz.GetSubstructMatches(_OH))
        d["broken_topology"] = not bz.HasSubstructMatch(
            Chem.MolFromSmarts("[#6][CX3](=[OX1])[CX4]([OX2H1])[#6,#1]")
        )
    return d


def robust_outliers(resid: np.ndarray, z: float):
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = 1.4826 * mad if mad > 0 else resid.std()
    mz = (resid - med) / scale
    return np.abs(mz) > z, med, scale, mz


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--z", type=float, default=3.5, help="modified-z outlier cutoff")
    args = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    for c in ("aldehyde_smiles", "benzoin_smiles", "dG_xtb_kcal", "dG_orca_kcal"):
        if c not in df.columns:
            raise SystemExit(f"missing column {c!r} in {args.input}")
    df["dG_xtb_kcal"] = pd.to_numeric(df["dG_xtb_kcal"], errors="coerce")
    df["dG_orca_kcal"] = pd.to_numeric(df["dG_orca_kcal"], errors="coerce")
    df = df.dropna(subset=["dG_xtb_kcal", "dG_orca_kcal"]).reset_index(drop=True)
    df["residual"] = df["dG_orca_kcal"] - df["dG_xtb_kcal"]  # DFT - xTB
    if "is_ewg" not in df.columns:
        df["is_ewg"] = df["aldehyde_smiles"].apply(is_ewg)
    have_decomp = {"D_react", "D_prod"}.issubset(df.columns)

    benign = df[~df["is_ewg"].astype(bool)].copy().reset_index(drop=True)
    n_b = len(benign)
    is_out, med, scale, mz = robust_outliers(benign["residual"].values, args.z)
    benign["is_outlier"] = is_out
    benign["mod_z"] = mz
    out = benign[benign["is_outlier"]].copy()
    bg = benign[~benign["is_outlier"]].copy()

    # descriptors for the benign set
    feats = [descriptors(a, b) for a, b in
             zip(benign["aldehyde_smiles"], benign["benzoin_smiles"])]
    fdf = pd.DataFrame(feats).reindex(benign.index)
    benign = pd.concat([benign, fdf], axis=1)
    out = benign[benign["is_outlier"]]
    bg = benign[~benign["is_outlier"]]

    # ── enrichment of binary features among outliers vs background ───────────────
    bin_feats = [c for c in fdf.columns
                 if fdf[c].dropna().isin([True, False]).all() and fdf[c].any()]
    rows = []
    for f in bin_feats:
        o = int(out[f].sum())
        b = int(bg[f].sum())
        fo = o / max(len(out), 1)
        fb = b / max(len(bg), 1)
        enr = (fo / fb) if fb > 0 else float("inf")
        rows.append((f, o, len(out), fo, b, len(bg), fb, enr))
    enr_df = pd.DataFrame(rows, columns=[
        "feature", "out_n", "out_tot", "out_frac",
        "bg_n", "bg_tot", "bg_frac", "enrichment"]).sort_values(
        "enrichment", ascending=False)

    # ── reactant vs product attribution ─────────────────────────────────────────
    attribution = ""
    if have_decomp and len(out):
        ps = out["D_prod"].abs() / (out["D_prod"].abs() + out["D_react"].abs())
        prod_dom = int((ps > 0.5).sum())
        attribution = (
            f"- Product-side dominated (|D_prod|>|D_react|): "
            f"**{prod_dom}/{len(out)}** ({100*prod_dom/len(out):.0f}%)\n"
            f"- Mean |D_react| = {out['D_react'].abs().mean():.1f} kcal, "
            f"mean |D_prod| = {out['D_prod'].abs().mean():.1f} kcal\n")

    # ── figures (one per file) ──────────────────────────────────────────────────
    figs = []
    plt.figure(figsize=(7, 5))
    plt.hist(bg["residual"], bins=60, color="#2c7fb8", alpha=0.8, label=f"benign inlier (n={len(bg)})")
    plt.hist(out["residual"], bins=30, color="#d7301f", alpha=0.8, label=f"benign outlier (n={len(out)})")
    for s in (med - args.z * scale, med + args.z * scale):
        plt.axvline(s, ls="--", c="k", lw=0.8)
    plt.xlabel("residual  ΔG_DFT − ΔG_xTB  [kcal/mol]"); plt.ylabel("count")
    plt.title(f"Non-EWG residual distribution (mod-z>{args.z} = outlier)")
    plt.legend(); plt.tight_layout()
    f1 = f"{outdir}/resid_hist_nonEWG_{ts}.png"; plt.savefig(f1, dpi=130); plt.close(); figs.append(f1)

    top = enr_df[(enr_df["out_n"] >= 2)].head(12)[::-1]
    if len(top):
        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"], top["enrichment"], color="#d7301f")
        plt.axvline(1.0, c="k", lw=0.8)
        plt.xlabel("enrichment (outlier frac / background frac)")
        plt.title(f"Structural features enriched among non-EWG outliers (n_out={len(out)})")
        plt.tight_layout()
        f2 = f"{outdir}/enrichment_bar_nonEWG_{ts}.png"; plt.savefig(f2, dpi=130); plt.close(); figs.append(f2)

    if "bz_heavy" in benign:
        plt.figure(figsize=(7, 5))
        plt.scatter(bg["bz_heavy"], bg["residual"], s=10, c="#2c7fb8", alpha=0.5, label="inlier")
        plt.scatter(out["bz_heavy"], out["residual"], s=18, c="#d7301f", label="outlier")
        plt.xlabel("benzoin heavy-atom count"); plt.ylabel("residual [kcal/mol]")
        plt.title("Non-EWG residual vs molecular size"); plt.legend(); plt.tight_layout()
        f3 = f"{outdir}/resid_vs_size_nonEWG_{ts}.png"; plt.savefig(f3, dpi=130); plt.close(); figs.append(f3)

    if "bz_nrot" in benign:
        plt.figure(figsize=(7, 5))
        plt.scatter(bg["bz_nrot"], bg["residual"], s=10, c="#2c7fb8", alpha=0.5, label="inlier")
        plt.scatter(out["bz_nrot"], out["residual"], s=18, c="#d7301f", label="outlier")
        plt.xlabel("benzoin rotatable bonds"); plt.ylabel("residual [kcal/mol]")
        plt.title("Non-EWG residual vs flexibility"); plt.legend(); plt.tight_layout()
        f4 = f"{outdir}/resid_vs_nrot_nonEWG_{ts}.png"; plt.savefig(f4, dpi=130); plt.close(); figs.append(f4)

    if have_decomp and len(out):
        plt.figure(figsize=(6, 6))
        plt.scatter(out["D_react"], out["D_prod"], s=20, c="#d7301f")
        lim = np.nanpercentile(np.abs(np.r_[out["D_react"], out["D_prod"]]), 99) * 1.1
        plt.plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
        plt.axhline(0, c="gray", lw=0.5); plt.axvline(0, c="gray", lw=0.5)
        plt.xlabel("D_react [kcal/mol]"); plt.ylabel("D_prod [kcal/mol]")
        plt.title("Non-EWG outliers: reactant vs product side disagreement")
        plt.tight_layout()
        f5 = f"{outdir}/Dreact_vs_Dprod_nonEWG_outliers_{ts}.png"; plt.savefig(f5, dpi=130); plt.close(); figs.append(f5)

    # ── outputs ─────────────────────────────────────────────────────────────────
    keep = [c for c in ["index", "PubChem_CID", "aldehyde_name", "aldehyde_smiles",
            "benzoin_smiles", "dG_xtb_kcal", "dG_orca_kcal", "residual", "mod_z",
            "D_react", "D_prod"] if c in out.columns] + bin_feats + \
           [c for c in ["bz_heavy", "bz_nrot", "bz_HBD", "bz_nOH"] if c in out.columns]
    out_sorted = out.reindex(out["residual"].abs().sort_values(ascending=False).index)
    csv_path = f"{outdir}/outliers_nonEWG_permol_{ts}.csv"
    out_sorted[keep].to_csv(csv_path, index=False)

    md = [f"# Non-EWG outliers of the xTB–DFT benzoin ΔG validation — {ts}", "",
          f"Source: `{args.input}`  ", f"Residual = ΔG_DFT − ΔG_xTB (positive = xTB over-exergonic).", "",
          "## Set sizes", "",
          f"- total molecules: {len(df)}", f"- EWG (excluded here): {int(df['is_ewg'].sum())}",
          f"- benign (non-EWG): {n_b}", f"- benign median residual: {med:.2f} kcal, robust σ (1.4826·MAD): {scale:.2f}",
          f"- **non-EWG outliers (|mod-z|>{args.z}): {len(out)}** ({100*len(out)/n_b:.1f}% of benign)", "",
          "## Reactant- vs product-side attribution", "",
          attribution or "_(no D_react/D_prod columns; run decomp_valid.py first for side attribution)_", "",
          "## Structural enrichment among non-EWG outliers", "",
          "| feature | outlier | (frac) | background | (frac) | enrichment |",
          "|---|---|---|---|---|---|"]
    for _, r in enr_df[enr_df["out_n"] >= 1].iterrows():
        md.append(f"| {r.feature} | {int(r.out_n)}/{int(r.out_tot)} | {r.out_frac:.2f} "
                  f"| {int(r.bg_n)}/{int(r.bg_tot)} | {r.bg_frac:.3f} | "
                  f"{r.enrichment:.1f}× |")
    md += ["", "## Worst 15 non-EWG outliers", "",
           "| residual | mod-z | aldehyde_smiles |", "|---|---|---|"]
    for _, r in out_sorted.head(15).iterrows():
        md.append(f"| {r.residual:+.1f} | {r.mod_z:+.1f} | `{r.aldehyde_smiles}` |")
    md += ["", "## Figures (standalone)", ""] + [f"- `{os.path.basename(f)}`" for f in figs]
    md_path = f"{outdir}/REPORT_outliers_nonEWG_{ts}.md"
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"benign non-EWG: {n_b}   outliers: {len(out)}")
    print(f"report  -> {md_path}")
    print(f"csv     -> {csv_path}")
    for f in figs:
        print(f"figure  -> {f}")
    print("\nTop enriched features (out_n>=2):")
    print(enr_df[enr_df["out_n"] >= 2].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
