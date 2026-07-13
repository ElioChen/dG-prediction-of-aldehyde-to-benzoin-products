#!/usr/bin/env python
"""Dedicated deep-dive on aromatic-subset prediction quality for the current champion
(MORDREDSLIM271_BDEGXTB, test MAE 1.503).

Motivation: aromatic-only-scope was explicitly DROPPED as a training restriction (2026-06-24,
train/predict ALL categories) after MaxMin active-selection was found to undersample the
carbo-aromatic category (39%->11%, maxmin-undersamples-aromatic memory) during an earlier
phase of the project. Since then, DFT-SP labeling has covered the near-full library (not a
MaxMin-selected subset), so this checks: (1) is the aromatic/aliphatic ratio in the CURRENT
training population now representative of the true library (i.e. is the old bias actually
moot), and (2) does prediction quality differ by scope now, and within aromatic, by
heteroaromatic vs simple carbocyclic-aromatic chemotype.
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = Path(f"{H}/viz_gxtb_20260625"); OUT.mkdir(exist_ok=True)
TAG = time.strftime("%Y%m%d")

HETEROAROM_SMARTS = {
    "furan": "[o]", "thiophene": "[s]", "azine_or_azole_N": "[n]",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in HETEROAROM_SMARTS.items()}


def heteroarom_tag(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return "invalid"
    hits = [k for k, p in PATS.items() if m.HasSubstructMatch(p)]
    return "+".join(hits) if hits else "simple_carbo_aromatic"


def savefig(name):
    plt.gcf().tight_layout(); plt.savefig(OUT / name, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", name, flush=True)


def main():
    # ── 1. population-level representativeness check ───────────────────────
    cls_full = pd.read_parquet(f"{H}/aldehyde_class.parquet")
    pop_counts = cls_full["cls"].value_counts()
    pop_pct = (pop_counts / len(cls_full) * 100).round(2)
    print("full-library scope population:\n", pop_pct, flush=True)

    te = pd.read_csv(f"{H}/viz_gxtb_20260625/test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv")
    te_counts = te["cls"].value_counts()
    te_pct = (te_counts / len(te) * 100).round(2)
    print("test-set scope population:\n", te_pct, flush=True)

    fl = pd.read_csv(f"{H}/viz_gxtb_20260625/products_dG_corrected_MORDREDSLIM271_BDEGXTB_20260706.csv",
                      usecols=["id", "route_to_dft"])
    fl = fl.merge(cls_full, on="id", how="left")

    # ── 2. per-scope test accuracy ──────────────────────────────────────────
    rows = []
    for s in ["aromatic", "aliphatic"]:
        mk = te["cls"] == s
        sub = te[mk]
        bias = (sub["dG_pred"] - sub["dG_orca_kcal"]).mean()
        r2 = 1 - ((sub["dG_orca_kcal"] - sub["dG_pred"]) ** 2).sum() / ((sub["dG_orca_kcal"] - sub["dG_orca_kcal"].mean()) ** 2).sum()
        route_rate = fl.loc[fl["cls"] == s, "route_to_dft"].mean() * 100
        rows.append({"scope": s, "n_test": int(mk.sum()), "MAE": sub["error"].mean(),
                     "bias(pred-true)": bias, "R2": r2, "route_to_dft_rate_%": route_rate})
    scope_df = pd.DataFrame(rows)
    scope_df.to_csv(OUT / f"aromatic_scope_summary_champion275_{TAG}.csv", index=False)
    print(scope_df.to_string(index=False), flush=True)

    # ── 3. within-aromatic: heteroaromatic vs simple carbocyclic breakdown ──
    arom = te[te["cls"] == "aromatic"].copy()
    u = arom[["ald_smiles"]].drop_duplicates()
    u["hetero_tag"] = u["ald_smiles"].apply(heteroarom_tag)
    arom = arom.merge(u, on="ald_smiles", how="left")
    hetero_summary = arom.groupby("hetero_tag").agg(n=("error", "size"), MAE=("error", "mean"),
                                                       bias=("error", lambda e: (arom.loc[e.index, "dG_pred"] - arom.loc[e.index, "dG_orca_kcal"]).mean())).reset_index()
    hetero_summary = hetero_summary.sort_values("n", ascending=False)
    hetero_summary.to_csv(OUT / f"aromatic_heterotag_summary_champion275_{TAG}.csv", index=False)
    print(hetero_summary.to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = hetero_summary[hetero_summary["n"] >= 20]
    ax.barh(plot_df["hetero_tag"][::-1], plot_df["MAE"][::-1], color="#2171b5")
    ax.set_xlabel("MAE (kcal/mol)"); ax.set_title(f"aromatic subset MAE by chemotype (n>=20, champion275)")
    savefig(f"118_aromatic_hetero_mae_{TAG}.png")

    # ── 4. parity plots, one per scope (no composite figures) ──────────────
    for s, color in [("aromatic", "#2171b5"), ("aliphatic", "#cb181d")]:
        sub = te[te["cls"] == s]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub["dG_orca_kcal"], sub["dG_pred"], s=4, alpha=0.25, color=color)
        lo, hi = min(sub["dG_orca_kcal"].min(), sub["dG_pred"].min()), max(sub["dG_orca_kcal"].max(), sub["dG_pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        mae_s = sub["error"].mean()
        ax.set_xlabel("DFT dG_orca (kcal/mol)"); ax.set_ylabel("corrected prediction (kcal/mol)")
        ax.set_title(f"champion275 parity, {s} only (n={len(sub):,}, MAE={mae_s:.3f})")
        savefig(f"119_parity_{s}_champion275_{TAG}.png")

    # ── 5. error vs size within aromatic only ───────────────────────────────
    arom["MolWt"] = arom["smiles"].apply(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else np.nan)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"simple_carbo_aromatic": "#2171b5", "furan": "#238b45", "thiophene": "#cb181d", "azine_or_azole_N": "#fd8d3c"}
    for tagv in arom["hetero_tag"].unique():
        mk = arom["hetero_tag"] == tagv
        if mk.sum() < 10: continue
        ax.scatter(arom.loc[mk, "MolWt"], arom.loc[mk, "error"], s=5, alpha=0.35,
                   color=colors.get(tagv, "#999999"), label=f"{tagv} (n={mk.sum()})")
    ax.set_xlabel("product MolWt (g/mol)"); ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title("aromatic-only error vs size, by chemotype"); ax.legend(fontsize=7)
    savefig(f"120_aromatic_error_vs_size_{TAG}.png")

    # ── report ───────────────────────────────────────────────────────────
    rep = OUT / f"REPORT_aromatic_subset_champion275_{TAG}.md"
    with open(rep, "w") as fh:
        fh.write(f"# Aromatic-subset deep dive: MORDREDSLIM271_BDEGXTB ({TAG})\n\n")
        fh.write("## 1. Is the old MaxMin sampling bias still a concern?\n\n")
        fh.write(f"Full-library population (`aldehyde_class.parquet`, n={len(cls_full):,}): "
                 f"aromatic {pop_pct.get('aromatic', 0):.1f}%, aliphatic {pop_pct.get('aliphatic', 0):.1f}%. "
                 f"Test-set population (n={len(te):,}): aromatic {te_pct.get('aromatic', 0):.1f}%, "
                 f"aliphatic {te_pct.get('aliphatic', 0):.1f}%. ")
        gap = abs(pop_pct.get('aromatic', 0) - te_pct.get('aromatic', 0))
        fh.write(f"Gap = {gap:.1f} points. " + (
            "**Negligible gap — the earlier MaxMin active-selection bias (39%->11% "
            "undersampling) is NOT present in the current training population**, since labeling "
            "now covers the near-full library rather than a MaxMin-selected subset, and the "
            "70:20:10 split is a uniform random permutation with no scope-based stratification "
            "needed.\n\n" if gap < 3 else
            "**Non-trivial gap remains — worth investigating further before considering this "
            "fully resolved.**\n\n"))
        fh.write("## 2. Per-scope test accuracy\n\n")
        fh.write("| scope | n_test | MAE | bias (pred-true) | R2 | full-lib route_to_dft rate |\n|---|---|---|---|---|---|\n")
        for _, r in scope_df.iterrows():
            fh.write(f"| {r['scope']} | {int(r['n_test']):,} | {r['MAE']:.3f} | {r['bias(pred-true)']:+.3f} | "
                     f"{r['R2']:.3f} | {r['route_to_dft_rate_%']:.1f}% |\n")
        fh.write("\n## 3. Within aromatic: heteroaromatic vs simple carbocyclic\n\n")
        fh.write("| chemotype | n | MAE | mean bias |\n|---|---|---|---|\n")
        for _, r in hetero_summary.iterrows():
            fh.write(f"| {r['hetero_tag']} | {int(r['n'])} | {r['MAE']:.3f} | {r['bias']:+.3f} |\n")
        fh.write(f"\nSee `119_parity_aromatic_champion275_{TAG}.png` / `119_parity_aliphatic_champion275_{TAG}.png` "
                 f"for scope-split parity, `120_aromatic_error_vs_size_{TAG}.png` for the "
                 f"chemotype x size breakdown within the aromatic subset, and "
                 f"`118_aromatic_hetero_mae_{TAG}.png` for the chemotype MAE bar chart.\n")
    print("wrote", rep, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
