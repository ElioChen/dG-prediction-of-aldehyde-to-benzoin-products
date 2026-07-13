#!/usr/bin/env python3
"""
Diagnostic analysis of the 1500-molecule xTB / r2SCAN-3c ΔG table
(data/featurize.parquet): distributions, the xTB→DFT correction, outliers, the
chemistry behind them, structural classes, and feature-space coverage of the
original-500 vs the extrapolated-1000. Drives the "reselect 2k / add 500"
decision.

Outputs
  runs/figs/energy_analysis.png         multi-panel diagnostic
  runs/data/energy_outliers.csv         flagged rows + reason (SMILES)
  runs/data/energy_analysis.json        machine-readable summary
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import delta_core as dc

REPO = dc.REPO_ROOT
PARQUET = REPO / "data/featurize.parquet"
XTB, ORCA = "dG_xtb_kcal", "dG_orca_kcal"

# Exotic / reactive groups xTB handles poorly and that are atypical for a benzoin
# aldehyde substrate. SMARTS → label.
EXOTIC = {
    "hypervalent_S(F5)": "[#16](F)(F)(F)F",
    "nitroso_N=O":       "[NX2]=[OX1]",
    "azo/diazo_N=N":     "[NX2]=[NX2]",
    "nitro":             "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "ketene/cumulene":   "[CX2]=[CX2]=[#6,#8]",
    "disulfide_S-S":     "[#16]-[#16]",
    "thiocarbonyl_C=S":  "[CX3]=[SX1]",
    "isocyanide/diazo":  "[$([CX1-]#[NX2+]),$([NX2]=[NX2+]=[#6-])]",
}


def flag_exotic(smiles: pd.Series):
    """Per-row list of exotic groups present (rdkit SMARTS); '' if clean."""
    from rdkit import Chem
    pats = {k: Chem.MolFromSmarts(v) for k, v in EXOTIC.items()}
    charged = Chem.MolFromSmarts("[+1,+2,-1,-2]")
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            out.append("unparseable"); continue
        hits = [k for k, p in pats.items() if p is not None and mol.HasSubstructMatch(p)]
        if mol.HasSubstructMatch(charged):
            hits.append("formal_charge")
        out.append(",".join(hits))
    return pd.Series(out, index=smiles.index)


def main() -> int:
    df = pd.read_parquet(PARQUET)
    orig = set(pd.read_csv(REPO / "data/library/subset_v2.csv")["index"])
    new = set(pd.read_csv(REPO / "data/library/subset_expansion_v3.csv")["index"])
    df["set"] = np.where(df["index"].isin(new), "new1000",
                         np.where(df["index"].isin(orig), "orig500", "other"))

    m = df[df[ORCA].notna() & df[XTB].notna()].copy()
    m["corr"] = m[ORCA] - m[XTB]
    med = m["corr"].median()
    mad = (m["corr"] - med).abs().median() or 1.0
    m["z"] = (m["corr"] - med) / (1.4826 * mad)            # robust z-score
    m["exotic"] = flag_exotic(m["SMILES"])
    m["is_exotic"] = m["exotic"].str.len() > 0

    # QC flags mirroring delta_core (the rows training drops).
    m["qc_fail"] = (m[XTB].abs() > 45) | (m[ORCA].abs() > 45) | (m["z"].abs() > 6 * 0.674)
    # (delta_core uses |corr-med| > 6·MAD ≡ |z| > 6·0.674 in robust-z units.)
    m["qc_fail"] = (m[XTB].abs() > 45) | (m[ORCA].abs() > 45) | ((m["corr"] - med).abs() > 6 * mad)

    n = len(m)
    summary = {
        "n_total_rows": int(len(df)), "n_labeled": n,
        "n_errored": int((df["error"].astype("string").fillna("") != "").sum()),
        "corr_median": float(med), "corr_mad": float(mad),
        "xtb_orca_pearson": float(m[XTB].corr(m[ORCA])),
        "n_qc_fail": int(m["qc_fail"].sum()),
        "n_exotic": int(m["is_exotic"].sum()),
        "exotic_qc_overlap": int((m["is_exotic"] & m["qc_fail"]).sum()),
    }
    print(f"labeled n={n}  corr median={med:.2f} (xTB underestimates ΔG)  "
          f"pearson(xtb,orca)={summary['xtb_orca_pearson']:.3f}")
    print(f"QC-fail rows: {summary['n_qc_fail']}  |  exotic-group rows: {summary['n_exotic']}"
          f"  |  exotic∩QC-fail: {summary['exotic_qc_overlap']}")

    print("\n--- exotic-group prevalence & correction spread ---")
    rows = []
    for grp in EXOTIC.keys() | {"formal_charge", "unparseable"}:
        sel = m["exotic"].str.contains(grp, regex=False)
        if sel.sum() == 0:
            continue
        g = m[sel]
        rows.append((grp, int(sel.sum()), float(g["corr"].std()),
                     int(g["qc_fail"].sum())))
    rows.sort(key=lambda r: -r[1])
    summary["exotic_groups"] = [dict(group=g, n=nn, corr_std=round(s, 1), qc_fail=q)
                                for g, nn, s, q in rows]
    for g, nn, s, q in rows:
        print(f"  {g:20s} n={nn:4d}  corr_std={s:5.1f}  qc_fail={q}")

    # correction by aromatic-ring count (classic benzoin substrate = 1 aromatic ring)
    print("\n--- correction by aromatic-ring count ---")
    by_ar = m.groupby(m["ArRings"].clip(upper=3))["corr"].agg(["count", "median", "std"])
    print(by_ar.round(2).to_string())
    summary["corr_by_arrings"] = {int(k): dict(n=int(v["count"]), med=round(v["median"], 2),
                                               std=round(v["std"], 2))
                                  for k, v in by_ar.iterrows()}

    # set comparison
    for s, g in m.groupby("set"):
        if s == "other":
            continue
        summary[f"set_{s}"] = dict(n=len(g), orca_std=round(g[ORCA].std(), 2),
                                   corr_med=round(g["corr"].median(), 2),
                                   qc_fail=int(g["qc_fail"].sum()),
                                   exotic=int(g["is_exotic"].sum()))

    # outlier CSV (QC-fail or exotic), worst-first
    out_dir = REPO / "runs/data"; out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO / "runs/figs"; fig_dir.mkdir(parents=True, exist_ok=True)
    outl = m[m["qc_fail"] | m["is_exotic"]].copy()
    outl = outl.reindex(outl["z"].abs().sort_values(ascending=False).index)
    outl[["index", "SMILES", "set", XTB, ORCA, "corr", "z", "MW",
          "exotic", "qc_fail"]].to_csv(out_dir / "energy_outliers.csv", index=False)
    print(f"\nWrote {len(outl)} flagged rows -> {out_dir/'energy_outliers.csv'}")

    # ── figure ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fig, ax = plt.subplots(3, 3, figsize=(17, 14))

    # (0,0) xTB vs ORCA parity, by set, QC-fail ringed
    a = ax[0, 0]
    for s, c in [("orig500", "C0"), ("new1000", "C1")]:
        g = m[m["set"] == s]
        a.scatter(g[XTB], g[ORCA], s=10, alpha=0.4, color=c, label=s)
    qf = m[m["qc_fail"]]
    a.scatter(qf[XTB], qf[ORCA], s=40, facecolors="none", edgecolors="r", label="QC-fail")
    lo, hi = m[XTB].min(), m[XTB].max()
    a.plot([lo, hi], [lo, hi], "k--", lw=1)
    a.set_xlabel("dG_xtb (kcal/mol)"); a.set_ylabel("dG_orca (kcal/mol)")
    a.set_title("xTB vs r2SCAN-3c ΔG"); a.legend(fontsize=8)

    # (0,1) zoom to physical window
    a = ax[0, 1]
    w = m[(m[XTB].between(-45, 45)) & (m[ORCA].between(-45, 45))]
    a.scatter(w[XTB], w[ORCA], s=10, alpha=0.4, c=w["corr"], cmap="coolwarm")
    a.plot([-45, 45], [-45, 45], "k--", lw=1)
    a.set_xlabel("dG_xtb"); a.set_ylabel("dG_orca")
    a.set_title("physical window (|ΔG|<45), colour=correction")

    # (0,2) distributions
    a = ax[0, 2]
    for col, c in [(XTB, "C0"), (ORCA, "C2")]:
        a.hist(m[col].clip(-45, 45), bins=60, alpha=0.5, label=col, color=c)
    a.set_xlabel("ΔG (kcal/mol)"); a.set_title("ΔG distributions (clipped ±45)")
    a.legend(fontsize=8)

    # (1,0) correction histogram
    a = ax[1, 0]
    a.hist(m["corr"].clip(-30, 50), bins=60, color="C3", alpha=0.7)
    a.axvline(med, color="k", ls="--", label=f"median {med:.1f}")
    a.set_xlabel("correction = orca − xtb"); a.set_title("Δ-learning target"); a.legend(fontsize=8)

    # (1,1) correction vs MW
    a = ax[1, 1]
    a.scatter(m["MW"], m["corr"], s=10, alpha=0.4, c=m["is_exotic"].map({True: "r", False: "C0"}))
    a.set_xlabel("MW"); a.set_ylabel("correction"); a.set_title("correction vs MW (red=exotic)")

    # (1,2) correction by aromatic ring count
    a = ax[1, 2]
    data = [m[m["ArRings"].clip(upper=3) == k]["corr"].clip(-30, 50) for k in range(4)]
    a.boxplot(data, tick_labels=["0", "1", "2", "3+"])
    a.set_xlabel("# aromatic rings"); a.set_ylabel("correction")
    a.set_title("correction by aromatic-ring count")

    # feature space: PCA + KMeans on the model features
    tbl = dc.load_training_table()                          # QC'd, imputed, scaled-ready
    Xs = StandardScaler().fit_transform(tbl.X.to_numpy())
    pca = PCA(n_components=2).fit(Xs); P = pca.transform(Xs)
    setq = df.set_index("index").loc[tbl.df["index"], "set"].to_numpy()
    corrq = tbl.dG_dft - tbl.dG_xtb

    # (2,0) PCA orig vs new
    a = ax[2, 0]
    for s, c in [("orig500", "C0"), ("new1000", "C1")]:
        sel = setq == s
        a.scatter(P[sel, 0], P[sel, 1], s=10, alpha=0.4, color=c, label=s)
    a.set_title(f"feature PCA ({pca.explained_variance_ratio_[:2].sum()*100:.0f}% var)")
    a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.legend(fontsize=8)

    # (2,1) PCA coloured by correction
    a = ax[2, 1]
    sc = a.scatter(P[:, 0], P[:, 1], s=10, alpha=0.5, c=corrq, cmap="coolwarm",
                   vmin=np.percentile(corrq, 2), vmax=np.percentile(corrq, 98))
    plt.colorbar(sc, ax=a, label="correction")
    a.set_title("PCA coloured by correction"); a.set_xlabel("PC1"); a.set_ylabel("PC2")

    # (2,2) KMeans clusters → mean |correction| (hardness map)
    a = ax[2, 2]
    k = 8
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    cl_corr = [np.abs(corrq[km.labels_ == i] - med).mean() for i in range(k)]
    cl_n = [int((km.labels_ == i).sum()) for i in range(k)]
    order = np.argsort(cl_corr)
    a.bar(range(k), [cl_corr[i] for i in order], color="C4")
    a.set_xticks(range(k)); a.set_xticklabels([f"c{i}\nn={cl_n[i]}" for i in order], fontsize=7)
    a.set_ylabel("mean |corr − median|"); a.set_title(f"KMeans(k={k}) cluster hardness")
    summary["cluster_hardness"] = sorted(
        [dict(cluster=int(i), n=cl_n[i], mean_abs_dev=round(float(cl_corr[i]), 2))
         for i in range(k)], key=lambda d: -d["mean_abs_dev"])

    fig.suptitle(f"Benzoin ΔG energy diagnostics — n={n} labeled "
                 f"(orig 500 + new 1000)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(fig_dir / "energy_analysis.png", dpi=130)
    print(f"Saved {fig_dir/'energy_analysis.png'}")

    json.dump(summary, open(out_dir / "energy_analysis.json", "w"), indent=2)
    print(f"Saved {out_dir/'energy_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
