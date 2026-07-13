#!/usr/bin/env python
"""aldehydes_clean_v6 / aldehydes_rejected_v6 — library composition visualization.

This is the STRUCTURAL library (filter_v6 output), not the ΔG screen: columns are
name/SMILES/formula/MW/InChIKey/CID/sources/cho_class/xtb_risk. Characterizes what the
production library contains and what filter_v6 threw out, per
[[filter-v3-relaxed]]/[[screen-v6-pipeline]]/[[aromatic-only-scope]].

Standalone figures (one per file, per no-composite-figures preference):
  1 MW distribution by cho_class (with the MW<=500 cap)
  2 cho_class composition — aromatic in-scope vs aliphatic out-of-scope
  3 xtb_risk tag breakdown (sulfonyl NOT tagged — known gap)
  4 filter_v6 rejection funnel (reject_reason)
"""
import os, datetime, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIB = "/scratch-shared/schen3/benzoin-dg/data/library"
OUT = "/scratch-shared/schen3/benzoin-dg/data/analysis/library_v6"
os.makedirs(OUT, exist_ok=True)
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CLS_COLOR = {"aromatic_carbo": "#3182bd", "aromatic_hetero": "#31a354", "aliphatic": "#bdbdbd"}
RED, ORANGE = "#d7301f", "#fdae6b"

def main():
    c = pd.read_csv(f"{LIB}/aldehydes_clean_v6.csv", low_memory=False)
    r = pd.read_csv(f"{LIB}/aldehydes_rejected_v6.csv", low_memory=False)
    c["MW"] = pd.to_numeric(c["MW"], errors="coerce")
    nC, nR = len(c), len(r)
    keep_rate = 100 * nC / (nC + nR)

    # ---- Fig 1: MW distribution by cho_class ----
    plt.figure(figsize=(7.6, 5.2))
    bins = np.linspace(0, 500, 76)
    for cls in ["aromatic_carbo", "aromatic_hetero", "aliphatic"]:
        sub = c.loc[c.cho_class == cls, "MW"].dropna()
        plt.hist(sub, bins=bins, histtype="step", lw=1.8, color=CLS_COLOR[cls],
                 label=f"{cls} (n={len(sub):,}, med {sub.median():.0f})")
    plt.axvline(500, color="k", ls="--", lw=1, label="filter_v6 cap MW=500")
    plt.xlabel("molecular weight [g/mol]"); plt.ylabel("count")
    plt.title(f"aldehydes_clean_v6 — MW landscape by carbonyl class (n={nC:,})")
    plt.legend(fontsize=8); plt.tight_layout()
    f1 = f"{OUT}/fig_lib_v6_MW_by_class_{TS}.png"; plt.savefig(f1, dpi=150); plt.close()

    # ---- Fig 2: cho_class composition (in-scope vs out) ----
    vc = c.cho_class.value_counts()
    order = ["aromatic_carbo", "aromatic_hetero", "aliphatic"]
    vals = [int(vc.get(k, 0)) for k in order]
    n_arom = vals[0] + vals[1]
    plt.figure(figsize=(7.2, 5.0))
    bars = plt.bar(order, vals, color=[CLS_COLOR[k] for k in order], width=0.62)
    for b, v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, v+1500, f"{v:,}\n({100*v/nC:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
    plt.axhspan(0, 0, color="none")
    plt.ylim(0, max(vals)*1.16)
    plt.ylabel("count")
    plt.title(f"aldehydes_clean_v6 composition by carbonyl class\n"
              f"aromatic in-scope {100*n_arom/nC:.1f}% ({n_arom:,}) · "
              f"aliphatic out-of-scope {100*vals[2]/nC:.1f}% ({vals[2]:,})",
              fontsize=11)
    plt.tight_layout()
    f2 = f"{OUT}/fig_lib_v6_cho_class_{TS}.png"; plt.savefig(f2, dpi=150); plt.close()

    # ---- Fig 3: xtb_risk breakdown (collapse multi-tags to each constituent) ----
    risk = c.xtb_risk.fillna("").astype(str).str.strip()
    n_clean = int((risk == "").sum())
    tags = ["nitro", "phosphorus", "boron", "selenium", "n_oxide"]
    tag_counts = {t: int(risk.str.contains(rf"\b{t}\b").sum()) for t in tags}
    n_multi = int((risk.str.contains(",")).sum())
    labels = ["(clean)"] + tags
    counts = [n_clean] + [tag_counts[t] for t in tags]
    plt.figure(figsize=(7.6, 5.0))
    cols = ["#9ecae1"] + [ORANGE]*len(tags)
    bars = plt.bar(labels, counts, color=cols, width=0.62)
    for b, v in zip(bars, counts):
        plt.text(b.get_x()+b.get_width()/2, v*1.02 if v>0 else 0,
                 f"{v:,}\n{100*v/nC:.2f}%", ha="center", va="bottom", fontsize=8)
    plt.yscale("log"); plt.ylim(1, n_clean*3)
    plt.ylabel("count (log)")
    plt.title("aldehydes_clean_v6 xtb_risk tags (kept-but-tagged set)\n"
              f"{n_clean:,} clean ({100*n_clean/nC:.1f}%) · {n_multi:,} carry >1 tag\n"
              "NOTE: sulfonyl / hypervalent-S is NOT tagged (known gap)",
              fontsize=10.5)
    plt.tight_layout()
    f3 = f"{OUT}/fig_lib_v6_xtb_risk_{TS}.png"; plt.savefig(f3, dpi=150); plt.close()

    # ---- Fig 4: filter_v6 rejection funnel ----
    rc = r.reject_reason.fillna("NA").value_counts()
    rc = rc.sort_values(ascending=True)
    plt.figure(figsize=(8.4, 6.2))
    y = np.arange(len(rc))
    plt.barh(y, rc.values, color="#969696")
    for yi, v in zip(y, rc.values):
        plt.text(v, yi, f" {v:,} ({100*v/nR:.1f}%)", va="center", fontsize=7)
    plt.yticks(y, rc.index, fontsize=8)
    plt.xlabel("molecules rejected")
    plt.xlim(0, rc.values.max()*1.18)
    plt.title(f"filter_v6 rejection funnel — {nR:,} rejected vs {nC:,} kept "
              f"(keep rate {keep_rate:.1f}%)")
    plt.tight_layout()
    f4 = f"{OUT}/fig_lib_v6_reject_funnel_{TS}.png"; plt.savefig(f4, dpi=150); plt.close()

    stats = dict(n_clean=nC, n_rejected=nR, keep_rate=keep_rate,
                 cho_class={k: int(vc.get(k,0)) for k in order},
                 n_aromatic_inscope=n_arom,
                 MW=dict(mean=float(c.MW.mean()), median=float(c.MW.median()),
                         min=float(c.MW.min()), max=float(c.MW.max())),
                 xtb_risk_clean=n_clean, xtb_risk_tags=tag_counts, xtb_risk_multi=n_multi,
                 reject_reason={k: int(v) for k, v in rc.sort_values(ascending=False).items()})
    json.dump(stats, open(f"{OUT}/stats_library_v6_{TS}.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print("wrote:", f1, f2, f3, f4, sep="\n  ")
    return stats

if __name__ == "__main__":
    main()
