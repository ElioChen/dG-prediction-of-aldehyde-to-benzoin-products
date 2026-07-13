#!/usr/bin/env python3
"""
Diagnose a featurize_screen run — characterise the failures and the physically
implausible ΔG outliers by functional group / element, quantify enrichment, and
render a multi-panel diagnostic FIGURE.

Reusable version of the filter_v5-pilot analysis: run it on any screen output
(pilot or the full library) to see WHICH chemistry breaks xTB before trusting the
distribution. Reads screen_all.csv if present (written by analyze_screen_v5.py),
otherwise concatenates <screen-dir>/chunk_*/features.csv.

Outputs:
  <screen-dir>/flagged.csv            every error row + every |ΔG|>cap row
  <out>/diagnostic.png                4-panel figure (failures / enrichment /
                                       ΔG-vs-size / worst outliers)

Usage:  python pipeline/diagnose_screen.py --screen-dir data/raw/screen_v5_pilot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parent.parent

# functional groups / elements to profile (label -> SMARTS)
GROUPS = {
    "B (boron)":   "[#5]",
    "Si":          "[#14]",
    "P":           "[#15]",
    "Se":          "[#34]",
    "nitro":       "[N+](=O)[O-]",
    "azide":       "[NX2,NX1-]=[NX2+]=[NX1-,NX2]",
    "N-oxide":     "[#7+][O-]",
    "3-ring":      "[r3]",
    "4-ring":      "[r4]",
    "macrocycle>12": "[r{13-}]",
    "perhalo-C":   "[CX4,c](F)(F)F",
}
_PATS = {k: Chem.MolFromSmarts(v) for k, v in GROUPS.items()}
ENRICH_GROUPS = ["B (boron)", "Si", "P", "Se", "nitro", "azide", "N-oxide",
                 "3-ring", "4-ring", "macrocycle>12", "perhalo-C"]

# risk-class priority for the ΔG-vs-size scatter (first match wins) + colour
RISK_CLASS = [
    ("malformed-B", "#000000"),
    ("boron",       "#C44E52"),
    ("P",           "#8172B2"),
    ("Se",          "#937860"),
    ("nitro",       "#DD8452"),
    ("N-oxide",     "#DA8BC3"),
    ("other",       "#BBBBBB"),
]
_RC_COLOR = dict(RISK_CLASS)


def _atoms_info(smi):
    """(nHeavy, risk_class, group-flag dict) for one SMILES, or (nan, 'other', {})."""
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return np.nan, "other", {}
    flags = {k: bool(p is not None and m.HasSubstructMatch(p)) for k, p in _PATS.items()}
    zs = {a.GetAtomicNum() for a in m.GetAtoms()}
    malformed_b = any(a.GetAtomicNum() == 5 and a.GetTotalValence() < 3 for a in m.GetAtoms())
    if malformed_b:                       rc = "malformed-B"
    elif 5 in zs:                         rc = "boron"
    elif 15 in zs:                        rc = "P"
    elif 34 in zs:                        rc = "Se"
    elif flags["nitro"]:                  rc = "nitro"
    elif flags["N-oxide"]:                rc = "N-oxide"
    else:                                 rc = "other"
    flags["MW"] = round(Descriptors.MolWt(m), 0)
    flags["nHeavy"] = m.GetNumHeavyAtoms()
    flags["maxRing"] = max((len(r) for r in m.GetRingInfo().AtomRings()), default=0)
    flags["malformed_B"] = malformed_b
    return m.GetNumHeavyAtoms(), rc, flags


def _load(screen_dir: Path) -> pd.DataFrame:
    sa = screen_dir / "screen_all.csv"
    if sa.exists():
        return pd.read_csv(sa)
    parts = sorted(screen_dir.glob("chunk_*/features.csv")) or sorted(screen_dir.glob("**/features.csv"))
    if not parts:
        raise SystemExit(f"No screen_all.csv or chunk_*/features.csv under {screen_dir}")
    return pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen-dir", required=True)
    ap.add_argument("--out", default=None, help="figure dir (default: <screen-dir>)")
    ap.add_argument("--dg-cap", type=float, default=30.0,
                    help="|dG_xtb| above this is a physically-implausible outlier")
    ap.add_argument("--soft-cap", type=float, default=20.0,
                    help="looser |dG| threshold for the enrichment comparison")
    ap.add_argument("--clean-sample", type=int, default=20000,
                    help="cap the clean-set baseline at N rows (speed; 0 = all)")
    ap.add_argument("--scatter-sample", type=int, default=20000,
                    help="cap the ΔG-vs-size scatter at N points")
    ap.add_argument("--show", type=int, default=20, help="rows to print per section")
    args = ap.parse_args()

    sd = Path(args.screen_dir)
    out_dir = Path(args.out) if args.out else sd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load(sd)
    dg = pd.to_numeric(df["dG_xtb_kcal"], errors="coerce")
    cho = df["cho_class"] if "cho_class" in df.columns else pd.Series(["?"] * len(df), index=df.index)
    err = df["error"].fillna("").astype(str)
    print(f"Loaded {len(df):,} rows  (dG present {int(dg.notna().sum()):,}, "
          f"errors {int((err.str.len()>0).sum()):,})")

    # ── A) Failures ──────────────────────────────────────────────────────────
    fails = df[err.str.len() > 0]
    fail_counts = err[err.str.len() > 0].value_counts()
    print(f"\n{'='*70}\nA) FAILURES ({len(fails):,})")
    if len(fails):
        print("  by reason:", dict(fail_counts))
        for _, r in fails.head(args.show).iterrows():
            _, _, f = _atoms_info(r["SMILES"])
            g = [k for k in GROUPS if f.get(k)]
            print(f"  {r['error']:18s} {str(cho[r.name]):15s} MW={f.get('MW')} "
                  f"nHeavy={f.get('nHeavy')} maxRing={f.get('maxRing')} grp={g}")

    # ── B) Implausible ΔG outliers ───────────────────────────────────────────
    out_mask = dg.abs() > args.dg_cap
    out = df[out_mask].assign(_dg=dg[out_mask]).sort_values("_dg")
    print(f"\n{'='*70}\nB) |ΔG|>{args.dg_cap:g} OUTLIERS ({len(out):,})")
    out_rows = []   # (dg, label, malformed_b) for the plot
    for _, r in out.iterrows():
        _, _, f = _atoms_info(r["SMILES"])
        g = [k for k in GROUPS if f.get(k)]
        out_rows.append((r["_dg"], f"{str(cho[r.name])[:4]} MW{int(f.get('MW',0))}", f.get("malformed_B", False)))
        if len(out_rows) <= args.show:
            mb = " MALFORMED-B" if f.get("malformed_B") else ""
            print(f"  ΔG={r['_dg']:>9.1f} {str(cho[r.name]):15s} MW={f.get('MW')} "
                  f"maxRing={f.get('maxRing')} grp={g}{mb}")
            print(f"      {str(r['SMILES'])[:75]}")

    # ── C) Enrichment: outlier (|ΔG|>soft) vs clean (|ΔG|<=soft) ──────────────
    hi = df[dg.abs() > args.soft_cap]
    lo_all = df[dg.abs() <= args.soft_cap]
    lo = lo_all.sample(args.clean_sample, random_state=0) if (args.clean_sample and len(lo_all) > args.clean_sample) else lo_all
    print(f"\n{'='*70}\nC) GROUP ENRICHMENT: |ΔG|>{args.soft_cap:g} (n={len(hi):,}) "
          f"vs clean (n={len(lo):,}{', sampled' if len(lo) < len(lo_all) else ''})")

    def rate(sub, k):
        p = _PATS[k]
        c = sum(1 for s in sub["SMILES"]
                if (lambda m: m is not None and m.HasSubstructMatch(p))(Chem.MolFromSmiles(s) if isinstance(s, str) else None))
        return c, len(sub)
    enrich = []   # (label, outlier%, clean%, ratio)
    for k in ENRICH_GROUPS:
        ch, nh = rate(hi, k); cl, nl = rate(lo, k)
        rh = ch / nh * 100 if nh else 0.0
        rl = cl / nl * 100 if nl else 0.0
        enrich.append((k, rh, rl, (rh / rl) if rl > 0 else np.nan))
        enr = f"{rh/rl:.1f}x" if rl > 0 else "—"
        print(f"  {k:16s} outlier {ch:4d}/{nh} ({rh:5.1f}%)   clean {cl:5d}/{nl} ({rl:5.2f}%)   {enr}")

    # ── flagged.csv ──────────────────────────────────────────────────────────
    flagged = df[(err.str.len() > 0) | out_mask].copy()
    flagged["flag"] = np.where(err[flagged.index].str.len() > 0, "error", "implausible_dG")
    cols = [c for c in ["index", "SMILES", "cho_class", "dG_xtb_kcal", "error", "flag"] if c in flagged.columns]
    flagged.to_csv(sd / "flagged.csv", columns=cols, index=False)
    print(f"\nWrote {sd/'flagged.csv'}  ({len(flagged):,} rows)")

    # ══ FIGURE ════════════════════════════════════════════════════════════════
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"featurize_screen diagnostic — {sd.name}  "
                 f"(N={len(df):,}; {len(fails)} fail, {int(out_mask.sum())} |ΔG|>{args.dg_cap:g})",
                 fontsize=13, fontweight="bold")

    # (A) failures by reason
    ax = axes[0, 0]
    if len(fail_counts):
        ax.barh(range(len(fail_counts)), fail_counts.values, color="#C44E52", alpha=0.85)
        ax.set_yticks(range(len(fail_counts))); ax.set_yticklabels(fail_counts.index, fontsize=8)
        ax.invert_yaxis()
        for i, v in enumerate(fail_counts.values):
            ax.text(v, i, f" {v}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no failures", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("count"); ax.set_title(f"(A) Failures by reason ({len(fails):,})")

    # (B) group enrichment (outlier% vs clean%)
    ax = axes[0, 1]
    labels = [e[0] for e in enrich]
    y = np.arange(len(labels))
    ax.barh(y + 0.2, [e[1] for e in enrich], height=0.4, color="#C44E52", alpha=0.85,
            label=f"|ΔG|>{args.soft_cap:g} outliers")
    ax.barh(y - 0.2, [e[2] for e in enrich], height=0.4, color="#4C72B0", alpha=0.85,
            label="clean")
    for i, e in enumerate(enrich):
        if not np.isnan(e[3]) and e[3] >= 2:
            ax.text(max(e[1], e[2]) + 0.3, i, f"{e[3]:.0f}×", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("% of molecules with group"); ax.set_title("(B) Group enrichment in ΔG outliers")
    ax.legend(fontsize=8)

    # (C) ΔG vs molecule size, coloured by risk class (sampled)
    ax = axes[1, 0]
    sdf = df[dg.notna()].copy(); sdf["_dg"] = dg[dg.notna()]
    if len(sdf) > args.scatter_sample:
        sdf = sdf.sample(args.scatter_sample, random_state=0)
    info = [_atoms_info(s) for s in sdf["SMILES"]]
    nheavy = np.array([t[0] for t in info], dtype=float)
    rcls = np.array([t[1] for t in info])
    for cls, color in RISK_CLASS:
        m = rcls == cls
        if m.any():
            ax.scatter(nheavy[m], sdf["_dg"].values[m], s=10 if cls != "other" else 5,
                       c=color, alpha=0.5 if cls == "other" else 0.85,
                       label=f"{cls} ({m.sum()})", zorder=1 if cls == "other" else 3,
                       edgecolors="none")
    # clip y to a readable window (extreme outliers go off-scale; panel D lists them)
    ylo, yhi = -(args.dg_cap + 15), args.dg_cap + 10
    n_off = int(((sdf["_dg"] < ylo) | (sdf["_dg"] > yhi)).sum())
    ax.set_ylim(ylo, yhi)
    ax.axhspan(args.dg_cap, yhi, color="red", alpha=0.06)
    ax.axhspan(ylo, -args.dg_cap, color="red", alpha=0.06)
    ax.axhline(args.dg_cap, ls="--", lw=0.7, color="red"); ax.axhline(-args.dg_cap, ls="--", lw=0.7, color="red")
    ax.set_xlabel("heavy atoms"); ax.set_ylabel("ΔG xTB (kcal/mol)")
    ax.set_title(f"(C) ΔG vs size — outliers shaded; coloured by risk class ({n_off} off-scale)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    # (D) worst |ΔG|>cap outliers
    ax = axes[1, 1]
    if out_rows:
        worst = sorted(out_rows, key=lambda t: abs(t[0]), reverse=True)[:15]
        vals = [w[0] for w in worst]; labs = [w[1] for w in worst]
        cols_d = ["#000000" if w[2] else "#937860" for w in worst]
        yb = range(len(worst))
        ax.barh(list(yb), vals, color=cols_d, alpha=0.85)
        ax.set_yticks(list(yb)); ax.set_yticklabels(labs, fontsize=7); ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.8)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
        ax.set_xlabel("ΔG xTB (kcal/mol)")
        ax.set_title("(D) Worst |ΔG| outliers (black = malformed-B)")
    else:
        ax.text(0.5, 0.5, "no |ΔG| outliers", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(D) Worst |ΔG| outliers")

    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic.png")
    plt.close(fig)
    print(f"Saved figure → {out_dir/'diagnostic.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
