#!/usr/bin/env python
"""screen_v6 — xtb_risk coverage-gap visualization (follow-up to funcgroup analysis).

Reuses the cached per-molecule group flags (screen_v6_groupflags_*.csv.gz) — no SMARTS
recompute. Quantifies the actionable finding from the funcgroup report and the DFT-SP
validation: the pipeline `xtb_risk` tag fires on nitro/B/P/Se/N-oxide but MISSES the
hypervalent-S / sulfonyl family (sulfonyl, sulfonyl fluoride, triflate) that the DFT-SP
validation proved is xTB's worst electronic failure. The sulfonyl family is also the
most enriched in the exergonic tail, so the screen's candidate ranking there is the
part the current tag protects least.

New standalone figures (one per file, per no-composite-figures preference):
  A) coverage of the EWG xTB-unreliable set: current xtb_risk vs proposed (+sulfonyl)
  B) suspect-fraction vs top-K most-exergonic candidates (selection "shopping list")
  C) tail composition: who occupies the most-exergonic 5%, tagged vs MISSED
"""
import os, glob, json, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/scratch-shared/schen3/benzoin-dg/data/analysis/screen_v6_funcgroups"
SRC = sorted(glob.glob(f"{BASE}/screen_v6_groupflags_*.csv.gz"))[-1]
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SULF = ["sulfonyl  S(=O)(=O)", "sulfonyl fluoride S(=O)2F", "triflate OS(=O)2CF3"]
RED, ORANGE, BLUE, GRAY = "#d7301f", "#fdae6b", "#3182bd", "#bdbdbd"

def main():
    df = pd.read_csv(SRC, low_memory=False)
    for c in ["any_ewg", "any_group"] + SULF + ["sulfonyl  S(=O)(=O)"]:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().isin(["True", "true", "1"])
    # robust bool cast for all flag cols
    flagcols = SULF + ["nitro [N+](=O)[O-]", "N-oxide", "boron", "phosphorus",
                       "selenium", "silicon", "any_ewg", "any_group"]
    for c in flagcols:
        df[c] = df[c].astype(str).str.strip().isin(["True", "true", "1"])
    df["dG"] = pd.to_numeric(df["dG"], errors="coerce")
    df = df.dropna(subset=["dG"]).reset_index(drop=True)

    # normalize current tag: non-empty xtb_risk string => currently flagged
    tag = df["xtb_risk"].fillna("").astype(str).str.strip().str.strip('"')
    df["tagged_now"] = tag.ne("") & tag.str.lower().ne("nan")
    df["has_sulf"] = df[SULF].any(axis=1)
    # proposed tag = current OR sulfonyl family
    df["tagged_proposed"] = df["tagged_now"] | df["has_sulf"]
    n = len(df)
    print(f"n={n:,}  any_ewg={int(df.any_ewg.sum()):,}  tagged_now={int(df.tagged_now.sum()):,} "
          f"has_sulf={int(df.has_sulf.sum()):,}  tagged_proposed={int(df.tagged_proposed.sum()):,}")

    # ---- Fig A: coverage of the EWG xTB-unreliable set ----
    ewg = df[df.any_ewg]
    n_ewg = len(ewg)
    cov_now = 100 * ewg.tagged_now.mean()
    cov_prop = 100 * ewg.tagged_proposed.mean()
    missed_now = int((~ewg.tagged_now).sum())
    missed_prop = int((~ewg.tagged_proposed).sum())
    plt.figure(figsize=(7.0, 5.2))
    bars = plt.bar(["current xtb_risk\n(nitro/B/P/Se/N-oxide)",
                    "proposed\n(+ sulfonyl family)"],
                   [cov_now, cov_prop], color=[ORANGE, RED], width=0.6)
    for b, miss in zip(bars, [missed_now, missed_prop]):
        h = b.get_height()
        inside = h > 90
        plt.text(b.get_x()+b.get_width()/2, h - 6 if inside else h + 1.5,
                 f"{h:.1f}%\ncovered\n({miss:,} missed)",
                 ha="center", va="top" if inside else "bottom", fontsize=9,
                 color="white" if inside else "black")
    plt.ylim(0, 112)
    plt.ylabel("% of EWG xTB-unreliable molecules flagged")
    plt.title(f"xtb_risk coverage of the DFT-proven xTB-unreliable set\n"
              f"(EWG-unreliable n={n_ewg:,}; adding sulfonyl family closes the gap)")
    plt.tight_layout()
    fA = f"{BASE}/fig_risk_coverage_gap_{TS}.png"; plt.savefig(fA, dpi=150); plt.close()

    # ---- Fig B: suspect-fraction vs top-K most-exergonic candidates ----
    s = df.sort_values("dG").reset_index(drop=True)   # most exergonic first
    Ks = np.unique(np.round(np.logspace(1, np.log10(n), 60)).astype(int))
    Ks = Ks[Ks >= 10]
    cum_ewg = np.cumsum(s.any_ewg.values.astype(float))
    cum_missed = np.cumsum((s.any_ewg.values & ~s.tagged_now.values).astype(float))
    frac_ewg = 100 * cum_ewg[Ks-1] / Ks
    frac_missed = 100 * cum_missed[Ks-1] / Ks
    plt.figure(figsize=(7.4, 5.2))
    plt.semilogx(Ks, frac_ewg, "-o", ms=3, color=RED,
                 label="carries EWG xTB-unreliable group")
    plt.semilogx(Ks, frac_missed, "-s", ms=3, color="#810f7c",
                 label="unreliable AND missed by current xtb_risk")
    plt.axhline(100*df.any_ewg.mean(), color="gray", ls="--", lw=1,
                label=f"library baseline {100*df.any_ewg.mean():.1f}%")
    plt.xlabel("top-K most-exergonic predicted candidates (ranked by ΔG xTB)")
    plt.ylabel("% of the top-K that are xTB-unreliable")
    plt.title("Picking the top-K screen hits draws disproportionately\n"
              "from molecules where xTB ΔG is least trustworthy")
    plt.legend(fontsize=8); plt.grid(alpha=0.25, which="both"); plt.tight_layout()
    fB = f"{BASE}/fig_topK_suspect_fraction_{TS}.png"; plt.savefig(fB, dpi=150); plt.close()

    # ---- Fig C: exergonic-tail composition (tagged vs missed) ----
    thr = np.percentile(df.dG, 5)
    tail = df[df.dG <= thr]
    nt = len(tail)
    n_clean = int((~tail.any_ewg).sum())
    n_caught = int((tail.any_ewg & tail.tagged_now).sum())
    n_missed = int((tail.any_ewg & ~tail.tagged_now).sum())
    plt.figure(figsize=(7.0, 5.2))
    segs = [("no EWG-unreliable group", n_clean, GRAY),
            ("EWG-unreliable, caught by xtb_risk", n_caught, ORANGE),
            ("EWG-unreliable, MISSED by xtb_risk", n_missed, RED)]
    left = 0
    for lab, v, c in segs:
        plt.barh(0, v, left=left, color=c, edgecolor="white",
                 label=f"{lab}  ({v:,} = {100*v/nt:.1f}%)")
        if v/nt > 0.03:
            plt.text(left+v/2, 0, f"{100*v/nt:.0f}%", ha="center", va="center",
                     fontsize=9, color="white" if c != GRAY else "black")
        left += v
    plt.xlim(0, nt); plt.yticks([])
    plt.xlabel(f"molecules in the most-exergonic 5% tail (ΔG ≤ {thr:.1f}, n={nt:,})")
    plt.title("Composition of the screen's exergonic tail:\n"
              f"{100*n_missed/nt:.0f}% are xTB-unreliable yet UNflagged by the current tag")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=8, frameon=False)
    plt.tight_layout()
    fC = f"{BASE}/fig_tail_composition_{TS}.png"; plt.savefig(fC, dpi=150,
                                                              bbox_inches="tight"); plt.close()

    stats = dict(n=n, n_ewg=n_ewg, cov_now=cov_now, cov_proposed=cov_prop,
                 missed_now=missed_now, missed_proposed=missed_prop,
                 tail_thr=float(thr), tail_n=nt, tail_clean=n_clean,
                 tail_caught=n_caught, tail_missed=n_missed,
                 tail_missed_pct=100*n_missed/nt)
    json.dump(stats, open(f"{BASE}/stats_risk_coverage_{TS}.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print("wrote:", fA, fB, fC, sep="\n  ")
    return stats

if __name__ == "__main__":
    main()
