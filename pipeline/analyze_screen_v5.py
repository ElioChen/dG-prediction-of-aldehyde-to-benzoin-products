#!/usr/bin/env python3
"""
Distribution analysis of the filter_v5 QM-descriptor + xTB-ΔG screen.

Concatenates the per-chunk outputs of featurize_screen.py
  <screen-dir>/chunk_*/features.csv
into one table, merges the filter_v5 `cho_class` back on by SMILES (the screen output
keeps SMILES but not cho_class), then plots the distributions of dG_xtb_kcal and the
key QM descriptors — overall and split by cho_class (aromatic_carbo / aromatic_hetero
/ aliphatic; v5 has no vinyl_conj).

Reads:  <screen-dir>/chunk_*/features.csv   (+ data/library/aldehydes_clean_v5.csv)
Writes: <screen-dir>/screen_all.csv         (concatenated, cho_class merged)
        data/analysis/screen_v5/*.png

Usage:  python pipeline/analyze_screen_v5.py --screen-dir data/raw/screen_v5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "data/library/aldehydes_clean_v5.csv"

C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#8C8C8C"]
CHO_COLORS = {"aromatic_carbo": C[0], "aromatic_hetero": C[1], "aliphatic": C[2]}
CHO_ORDER = ["aromatic_carbo", "aromatic_hetero", "aliphatic"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# descriptor columns to profile (label, column) — skip ones that are all-empty
DESCRIPTORS = [
    ("ΔG xTB (kcal/mol)", "dG_xtb_kcal"),
    ("xTB gap (eV)",      "xtb_gap"),
    ("xTB ω electrophil.", "xtb_omega"),
    ("xTB dipole (D)",    "xtb_dipole"),
    ("Mulliken q(CHO C)", "mulliken_CHO_C"),
    ("Fukui f+ (CHO C)",  "fukui_plus_CHO_C"),
    ("dual descr. (CHO C)", "dual_descriptor_CHO_C"),
    ("WBO C=O",           "wbo_CO"),
    ("Proton aff. O (kcal)", "pa_CHO_O"),
    ("%Vbur (CHO C)",     "vbur_CHO_C"),
    ("Sterimol L (Å)",    "sterimol_L"),
    ("SASA (Å²)",         "SASA_total"),
]


def _fmt(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen-dir", required=True,
                    help="dir containing chunk_*/features.csv from featurize_screen.py")
    ap.add_argument("--lib", default=str(LIB), help="filter_v5 clean CSV (for cho_class)")
    ap.add_argument("--out", default=str(REPO / "data/analysis/screen_v5"))
    ap.add_argument("--dg-cap", type=float, default=30.0,
                    help="|dG_xtb| above this (kcal/mol) is flagged physically "
                         "implausible (xTB artifact) and excluded from stats/plots")
    args = ap.parse_args()

    screen_dir = Path(args.screen_dir)
    parts = sorted(screen_dir.glob("chunk_*/features.csv"))
    if not parts:
        parts = sorted(screen_dir.glob("**/features.csv"))
    if not parts:
        raise SystemExit(f"No chunk_*/features.csv found under {screen_dir}")
    df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    print(f"Concatenated {len(parts)} chunks -> {len(df):,} rows")

    # Merge cho_class from the v5 library on SMILES (screen output keeps SMILES).
    if Path(args.lib).exists():
        lib = pd.read_csv(args.lib, usecols=["SMILES", "cho_class"]).drop_duplicates("SMILES")
        df = df.merge(lib, on="SMILES", how="left")
        n_cls = df["cho_class"].notna().sum()
        print(f"Merged cho_class for {n_cls:,}/{len(df):,} rows")
    else:
        df["cho_class"] = np.nan
        print(f"WARNING: library {args.lib} not found — cho_class unavailable")

    # |dG_xtb| sanity filter: flag physically-impossible xTB artifacts (persisted
    # as a column so downstream selection/modeling can drop them trivially).
    dg = pd.to_numeric(df["dG_xtb_kcal"], errors="coerce")
    df["dg_implausible"] = dg.abs() > args.dg_cap
    ok = dg.notna() & ~df["dg_implausible"]          # plausible ΔG rows

    out_csv = screen_dir / "screen_all.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # ── Summary ───────────────────────────────────────────────────────────────
    has_err = df["error"].fillna("").astype(str).str.len() > 0
    n_dg = int(dg.notna().sum())
    n_imp = int(df["dg_implausible"].sum())
    print(f"\n{'='*60}")
    print(f"Total rows        : {len(df):,}")
    print(f"With dG_xtb       : {n_dg:,} ({n_dg/len(df)*100:.1f}%)")
    print(f"With errors       : {int(has_err.sum()):,}")
    print(f"Implausible ΔG    : {n_imp:,}  (|ΔG|>{args.dg_cap:g}; flagged dg_implausible, excluded below)")
    if has_err.any():
        print("Top error reasons :")
        for r, n in df.loc[has_err, "error"].value_counts().head(8).items():
            print(f"  {n:7,d}  {r}")
    if df["cho_class"].notna().any():
        print(f"By cho_class (plausible ΔG only):")
        for k in CHO_ORDER:
            v = dg[ok & (df.cho_class == k)]
            if len(v):
                print(f"  {k:16s}: n={len(v):,}  ΔG median={v.median():+.2f}  "
                      f"mean={v.mean():+.2f}  [{v.min():+.1f},{v.max():+.1f}]")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Fig 1: ΔG xTB distribution (overall + by cho_class) ──────────────────
    # A small fraction of molecules give physically impossible xTB ΔG (e.g. < -100
    # kcal/mol — bad geometry / xTB artifact on exotic structures). Clip the DISPLAY
    # to a plausible window so the real peak is readable; report how many are excluded.
    dg = pd.to_numeric(df["dG_xtb_kcal"], errors="coerce")
    lo, hi = dg.quantile(0.005), dg.quantile(0.995)
    lo, hi = min(lo, -30), max(hi, 20)            # never tighter than [-30, 20]
    inwin = dg[(dg >= lo) & (dg <= hi)]
    n_out = int(dg.notna().sum() - inwin.notna().sum())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"xTB ΔG of benzoin condensation — filter_v5 screen "
                 f"(N={int(dg.notna().sum()):,}; {n_out} off-scale |ΔG| outliers hidden)",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.hist(inwin.dropna(), bins=80, range=(lo, hi), color=C[0], alpha=0.85, edgecolor="none")
    ax.axvline(0, ls="--", color="black", lw=0.9)
    ax.axvline(dg.median(), ls="-", color=C[3], lw=1.2, label=f"median {dg.median():+.2f}")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("ΔG xTB (kcal/mol)"); ax.set_ylabel("Count")
    ax.set_title("Overall (negative = favourable)"); ax.legend(); _fmt(ax)
    ax = axes[1]
    for k in CHO_ORDER:
        v = dg[(df.cho_class == k) & (dg >= lo) & (dg <= hi)].dropna()
        if len(v) > 10:
            ax.hist(v, bins=70, range=(lo, hi), color=CHO_COLORS[k], alpha=0.55,
                    density=True, label=f"{k} ({len(v):,})", edgecolor="none")
    ax.axvline(0, ls="--", color="black", lw=0.9)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("ΔG xTB (kcal/mol)"); ax.set_ylabel("Density")
    ax.set_title("By cho_class (normalised)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "01_dG_distribution.png"); plt.close(fig)
    print(f"Saved: {out_dir/'01_dG_distribution.png'}")

    # ── Fig 2: QM descriptor distributions grid ──────────────────────────────
    avail = [(lbl, col) for lbl, col in DESCRIPTORS
             if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()]
    ncol = 3
    nrow = (len(avail) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle("QM descriptor distributions — filter_v5 screen", fontsize=13, fontweight="bold")
    for ax, (lbl, col) in zip(axes, avail):
        v = pd.to_numeric(df[col], errors="coerce")
        lo, hi = v.quantile(0.005), v.quantile(0.995)   # clip extreme tails for readability
        ax.hist(v.clip(lo, hi).dropna(), bins=60, color=C[0], alpha=0.85, edgecolor="none")
        ax.set_title(f"{lbl}  (med={v.median():.2f})", fontsize=9)
        ax.set_ylabel("Count"); _fmt(ax)
    for ax in axes[len(avail):]:
        ax.set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "02_descriptor_distributions.png"); plt.close(fig)
    print(f"Saved: {out_dir/'02_descriptor_distributions.png'}")

    print(f"\nAll plots → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
