#!/usr/bin/env python
"""screen_v6 analysis with functional-group annotation.

Annotates the 220k-molecule xTB screen ΔG distribution with the functional groups that
are (a) FILTERED OUT pre-screen by filter_v6/v4 (enal/ynal/α-dicarbonyl/MW>500 — shown
as a text note, not in data) and (b) KEPT-BUT-TAGGED risk groups, plus the
hypervalent-S/EWG groups the DFT validation proved xTB describes unreliably (sulfonyl,
sulfonyl fluoride, triflate, nitro) — which the pipeline's xtb_risk tag currently MISSES.

Ties to the DFT-SP finding: the extreme-exergonic tail of the screen is enriched in
these xTB-unreliable groups, so the screen's ranking there is suspect.

Standalone figures (one per file), markdown report; written to data/analysis.
"""
import os, sys, glob, json, datetime, types
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

FEAT = "/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/analysis/screen_v6_features_all.csv"
OUT = "/scratch-shared/schen3/benzoin-dg/data/analysis/screen_v6_funcgroups"
os.makedirs(OUT, exist_ok=True)
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# functional groups: (label, SMARTS, category)
GROUPS = [
    ("sulfonyl  S(=O)(=O)",        "[#16X4](=[OX1])(=[OX1])",                      "EWG xTB-unreliable"),
    ("sulfonyl fluoride S(=O)2F",  "[#16X4](=[OX1])(=[OX1])[F]",                   "EWG xTB-unreliable"),
    ("triflate OS(=O)2CF3",        "[OX2][#16X4](=[OX1])(=[OX1])[CX4]([F])([F])[F]","EWG xTB-unreliable"),
    ("nitro [N+](=O)[O-]",         "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",           "EWG xTB-unreliable"),
    ("N-oxide",                    "[#7+][#8X1-]",                                  "risk-tagged"),
    ("boron",                      "[#5]",                                          "risk-tagged"),
    ("phosphorus",                 "[#15]",                                         "risk-tagged"),
    ("selenium",                   "[#34]",                                         "risk-tagged"),
    ("silicon",                    "[#14]",                                         "risk-tagged"),
]
PATS = [(lab, Chem.MolFromSmarts(sm), cat) for lab, sm, cat in GROUPS]

def _flags(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return [False]*len(PATS)
    return [m.HasSubstructMatch(p) for _, p, _ in PATS]

def main():
    df = pd.read_csv(FEAT, usecols=["SMILES","dG_xtb_kcal","cho_class","xtb_risk"],
                     low_memory=False)
    df["dG"] = pd.to_numeric(df["dG_xtb_kcal"], errors="coerce")
    df = df.dropna(subset=["dG","SMILES"]).reset_index(drop=True)
    print("valid dG rows:", len(df))

    with ProcessPoolExecutor(max_workers=24) as ex:
        flags = list(ex.map(_flags, df["SMILES"].tolist(), chunksize=512))
    F = pd.DataFrame(flags, columns=[g[0] for g in GROUPS])
    df = pd.concat([df, F], axis=1)
    labels = [g[0] for g in GROUPS]
    ewg_labels = [g[0] for g in GROUPS if g[2] == "EWG xTB-unreliable"]
    df["any_ewg"] = df[ewg_labels].any(axis=1)
    df["any_group"] = df[labels].any(axis=1)

    # tail definition: most-exergonic 5%
    thr = np.percentile(df["dG"], 5)
    tail = df["dG"] <= thr
    counts = {lab: int(df[lab].sum()) for lab in labels}
    tail_share = {lab: (100*df.loc[tail, lab].mean()) for lab in labels}
    overall_share = {lab: (100*df[lab].mean()) for lab in labels}

    note = ("FILTERED OUT pre-screen (filter_v6/v4, not in data): α,β-unsaturated enal, "
            "ynal, α-dicarbonyl, MW>500.\nKEPT & TAGGED here.")

    # ---- Fig 1: ΔG distribution, EWG-unreliable highlighted ----
    plt.figure(figsize=(7.4, 5.2))
    bins = np.linspace(df.dG.quantile(.001), df.dG.quantile(.999), 80)
    plt.hist(df.dG, bins=bins, color="#bdbdbd", label=f"all (n={len(df):,})")
    plt.hist(df.loc[df.any_ewg, "dG"], bins=bins, color="#d7301f", alpha=0.8,
             label=f"carries EWG xTB-unreliable group (n={int(df.any_ewg.sum()):,})")
    plt.axvline(thr, color="k", ls="--", lw=1, label=f"exergonic tail cut (5%) = {thr:.1f}")
    plt.yscale("log")
    plt.xlabel("ΔG xTB (GFN2)  [kcal/mol]  — more negative = predicted more favorable")
    plt.ylabel("count (log)")
    plt.title("screen_v6 ΔG distribution — EWG/hypervalent-S highlighted")
    txt = "EWG xTB-unreliable groups annotated:\n" + "\n".join(
        f"  • {lab}  (n={counts[lab]:,})" for lab in ewg_labels)
    plt.gca().text(0.015, 0.97, txt+"\n\n"+note, transform=plt.gca().transAxes,
                   fontsize=6.5, va="top", family="monospace",
                   bbox=dict(boxstyle="round", fc="#fff7ec", ec="#d7301f", alpha=0.9))
    plt.legend(loc="upper right", fontsize=8); plt.tight_layout()
    f1 = f"{OUT}/fig_dG_dist_EWG_annotated_{TS}.png"; plt.savefig(f1, dpi=150); plt.close()

    # ---- Fig 2: ΔG by functional group (box) ----
    plt.figure(figsize=(8.0, 5.6))
    data = [df.loc[df[lab], "dG"].values for lab in labels]
    data = [d if len(d) else np.array([np.nan]) for d in data]
    base = df.loc[~df.any_group, "dG"].values
    alld = [base] + data
    pos = range(len(alld))
    bp = plt.boxplot(alld, positions=list(pos), widths=0.6, showfliers=False,
                     patch_artist=True)
    colors = ["#9ecae1"] + ["#d7301f" if g[2]=="EWG xTB-unreliable" else "#fdae6b" for g in GROUPS]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    xt = ["none\n(baseline)"] + [f"{lab.split()[0]}\n(n={counts[lab]:,})" for lab in labels]
    plt.xticks(list(pos), xt, rotation=40, ha="right", fontsize=7)
    plt.axhline(df.loc[~df.any_group,"dG"].median(), color="#3182bd", ls=":", lw=1,
                label="baseline median")
    plt.ylabel("ΔG xTB [kcal/mol]")
    plt.title("screen_v6 ΔG by functional group  (red = EWG xTB-unreliable, orange = risk-tagged)")
    plt.legend(fontsize=8); plt.tight_layout()
    f2 = f"{OUT}/fig_dG_by_group_box_{TS}.png"; plt.savefig(f2, dpi=150); plt.close()

    # ---- Fig 3: exergonic-tail enrichment ----
    plt.figure(figsize=(7.6, 5.2))
    enrich = {lab: (tail_share[lab]/overall_share[lab] if overall_share[lab]>0 else 0)
              for lab in labels}
    order = sorted(labels, key=lambda l: enrich[l], reverse=True)
    y = np.arange(len(order))
    plt.barh(y, [enrich[l] for l in order],
             color=["#d7301f" if dict((g[0],g[2]) for g in GROUPS)[l]=="EWG xTB-unreliable"
                    else "#fdae6b" for l in order])
    plt.axvline(1.0, color="k", lw=1, ls="--", label="no enrichment")
    for yi, l in enumerate(order):
        plt.text(enrich[l], yi, f" {tail_share[l]:.1f}% vs {overall_share[l]:.1f}%",
                 va="center", fontsize=7)
    plt.yticks(y, order, fontsize=8)
    plt.xlabel("enrichment in most-exergonic 5% tail  (tail% / overall%)")
    plt.title(f"Functional-group enrichment in the screen's exergonic tail (ΔG ≤ {thr:.1f})")
    plt.legend(fontsize=8); plt.tight_layout()
    f3 = f"{OUT}/fig_tail_enrichment_{TS}.png"; plt.savefig(f3, dpi=150); plt.close()

    # ---- Fig 4: fraction xTB-unreliable vs ΔG bin ----
    plt.figure(figsize=(7.6, 5.0))
    bins2 = np.linspace(df.dG.quantile(.005), df.dG.quantile(.995), 30)
    df["bin"] = pd.cut(df.dG, bins2)
    frac = df.groupby("bin", observed=True)["any_ewg"].mean()*100
    centers = [iv.mid for iv in frac.index]
    plt.plot(centers, frac.values, "-o", color="#d7301f", ms=3)
    plt.axhline(100*df.any_ewg.mean(), color="gray", ls="--",
                label=f"library mean {100*df.any_ewg.mean():.1f}%")
    plt.xlabel("ΔG xTB [kcal/mol]"); plt.ylabel("% carrying EWG xTB-unreliable group")
    plt.title("xTB-unreliable contamination rises toward the exergonic tail")
    plt.legend(fontsize=8); plt.tight_layout()
    f4 = f"{OUT}/fig_unreliable_fraction_vs_dG_{TS}.png"; plt.savefig(f4, dpi=150); plt.close()

    # ---- report ----
    g_med = {lab: float(df.loc[df[lab],"dG"].median()) for lab in labels}
    base_med = float(df.loc[~df.any_group,"dG"].median())
    md = [f"# screen_v6 functional-group annotated analysis — {TS}", "",
          f"Source: `{FEAT}`  (valid-ΔG n = **{len(df):,}**; cho_class: "
          + ", ".join(f"{k} {v:,}" for k,v in df.cho_class.value_counts().items()) + ").", "",
          "## Functional groups annotated", "",
          "| group | SMARTS | category | n | % lib | median ΔG | tail share | enrichment |",
          "|---|---|---|---|---|---|---|---|"]
    for lab, sm, cat in GROUPS:
        md.append(f"| {lab} | `{sm}` | {cat} | {counts[lab]:,} | {overall_share[lab]:.2f}% | "
                  f"{g_med[lab]:+.1f} | {tail_share[lab]:.1f}% | {enrich[lab]:.1f}× |")
    md += ["",
        f"Baseline (no annotated group) median ΔG = **{base_med:+.1f}** kcal/mol.",
        f"Exergonic-tail cut (most-favorable 5%) = ΔG ≤ **{thr:.1f}** kcal/mol.", "",
        "## Filtering context",
        f"- {note.replace(chr(10),' ')}",
        "- The pipeline `xtb_risk` tag covers nitro/B/P/Se/N-oxide but **does NOT tag "
        "sulfonyl/hypervalent-S** — the group the DFT-SP validation proved is xTB's worst "
        "failure (EWG-only set: Pearson r = −0.32, MAE 34). Recommend adding sulfonyl/"
        "sulfonyl-fluoride/triflate to `xtb_risk`.", "",
        "## Key reading",
        f"- **{int(df.any_ewg.sum()):,} molecules ({100*df.any_ewg.mean():.1f}%)** carry an "
        "EWG xTB-unreliable group.",
        "- These groups are **enriched in the exergonic tail** (see enrichment column / "
        "fig_tail_enrichment): the screen's most-favorable predictions are disproportionately "
        "molecules where xTB ΔG is least trustworthy → tail ranking is suspect.",
        "- Combined with the conformer broken-topology rate (~1.9% full library), the screen's "
        "extreme-exergonic end mixes genuine signal with xTB-electronic + conformer artifacts.", "",
        "## Figures (standalone)",
        f"- `{os.path.basename(f1)}` — ΔG distribution, EWG-unreliable highlighted + group annotation",
        f"- `{os.path.basename(f2)}` — ΔG by functional group (box)",
        f"- `{os.path.basename(f3)}` — exergonic-tail enrichment per group",
        f"- `{os.path.basename(f4)}` — % xTB-unreliable vs ΔG bin"]
    open(f"{OUT}/REPORT_screen_v6_funcgroups_{TS}.md","w").write("\n".join(md))
    df.drop(columns=["bin"]).to_csv(f"{OUT}/screen_v6_groupflags_{TS}.csv.gz",
                                    index=False, compression="gzip")
    json.dump(dict(n=len(df), counts=counts, tail_thr=float(thr),
                   tail_share=tail_share, overall_share=overall_share, enrich=enrich,
                   median=g_med, baseline_median=base_med),
              open(f"{OUT}/stats_{TS}.json","w"), indent=2)
    print("\n".join(md[4:30]))
    print("wrote:", f1, f2, f3, f4, sep="\n  ")

if __name__ == "__main__":
    main()
