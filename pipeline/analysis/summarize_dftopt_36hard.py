#!/usr/bin/env python
"""Summarize the fair DFT-opt rerun on the 36 g-xTB-converged hard cases (job 24299886).

Reads dftopt_36hard/row_*.csv (ald_conv/bz_conv/note) + orca_out/*_opt.out, joins the
hardclass from dftopt_36hard_select.csv, reports DFT-opt convergence per class, and greps
the ORCA outputs for the literal failure string (SCF non-convergence vs other).
Writes a dated markdown report. One file; never overwrites.
"""
import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/scratch-shared/schen3/benzoin-dg")
RES = ROOT / "data/raw/screen_v6/dft_sp_r2scan3c/dftopt_36hard"
SEL = ROOT / "data/raw/screen_v6/dft_sp_r2scan3c/analysis/dftopt_36hard_select.csv"
STAMP = "20260629"
REP = RES / f"REPORT_dftopt_36hard_{STAMP}.md"

ERR_PATTERNS = [
    ("SCF_NOT_CONVERGED", r"SCF NOT CONVERGED|This wavefunction IS NOT CONVERGED|"
                          r"CONVERGENCE FAILURE|SERIOUS PROBLEM IN SOSCF"),
    ("GEOM_NOT_CONVERGED", r"The optimization did not converge|OPTIMIZATION RUN DONE.*not"),
    ("ORCA_ERROR_GENERIC", r"ORCA finished by error|aborting the run|"
                           r"TERMINATING THE PROGRAM|Error \(ORCA"),
]


def scan_orca_out(p: Path):
    try:
        t = p.read_text(errors="ignore")
    except Exception:
        return "no_file", False
    converged = "THE OPTIMIZATION HAS CONVERGED" in t
    hits = [name for name, pat in ERR_PATTERNS if re.search(pat, t)]
    return ("|".join(hits) if hits else ("converged" if converged else "unknown")), converged


def main():
    sel = pd.read_csv(SEL)
    cls = dict(zip(sel["idx"].astype(str), sel["hardclass"]))

    rows = []
    for f in sorted(glob.glob(str(RES / "row_*.csv"))):
        d = pd.read_csv(f)
        if len(d):
            rows.append(d.iloc[0].to_dict())
    if not rows:
        REP.write_text(f"# DFT-opt 36-hard rerun — NO RESULTS YET ({STAMP})\n\n"
                       "No row_*.csv found. Job 24299886 may still be running.\n")
        print("no results yet"); return
    df = pd.DataFrame(rows)
    df["hardclass"] = df["idx"].astype(str).map(cls).fillna("?")
    df["both_conv"] = df.get("ald_conv", False).astype(str).eq("True") & \
                      df.get("bz_conv", False).astype(str).eq("True")

    n = len(df)
    n_both = int(df["both_conv"].sum())
    n_ald = int(df.get("ald_conv").astype(str).eq("True").sum())
    n_bz = int(df.get("bz_conv").astype(str).eq("True").sum())

    # ORCA out diagnosis
    diag = {}
    for p in sorted(glob.glob(str(RES / "orca_out" / "*_opt.out"))):
        label, conv = scan_orca_out(Path(p))
        diag[Path(p).name] = label
    from collections import Counter
    diag_counts = Counter(diag.values())

    by_class = df.groupby("hardclass")["both_conv"].agg(["sum", "count"])

    L = [f"# Fair DFT-opt rerun on the 36 g-xTB-hard cases ({STAMP})\n",
         f"Job 24299886 · r2SCAN-3c Opt TightSCF CPCM(DMSO) · same 36 that g-xTB-opt did 36/36.\n\n",
         f"**Results in: {n}/36 molecules.**\n\n",
         f"- DFT-opt converged on BOTH aldehyde+product: **{n_both}/{n}**\n",
         f"- aldehyde converged: {n_ald}/{n}; product converged: {n_bz}/{n}\n\n",
         "## Convergence by hard class\n",
         "| class | both-converged / total |\n|---|---|\n"]
    for c, r in by_class.iterrows():
        L.append(f"| {c} | {int(r['sum'])}/{int(r['count'])} |\n")
    L.append("\n## ORCA opt failure mode (from retained input.out)\n")
    if diag_counts:
        for k, v in diag_counts.most_common():
            L.append(f"- `{k}`: {v}\n")
    else:
        L.append("- (no orca_out/*.out captured)\n")
    L.append("\n## Head-to-head (matched 36)\n")
    L.append(f"| method | geometry-opt convergence |\n|---|---|\n")
    L.append(f"| g-xTB-opt | 36/36 (gxtb_opt_moremols_20260620.csv) |\n")
    L.append(f"| DFT-opt (r2SCAN-3c) | **{n_both}/{n}** (this run) |\n\n")
    L.append("## Per-molecule notes\n")
    for _, r in df.sort_values("hardclass").iterrows():
        L.append(f"- idx {r['idx']} [{r['hardclass']}] ald_conv={r.get('ald_conv')} "
                 f"bz_conv={r.get('bz_conv')} note={r.get('note')}\n")
    REP.write_text("".join(L))
    print("wrote", REP)
    print(f"DFT-opt both-converged: {n_both}/{n}  | diag: {dict(diag_counts)}")


if __name__ == "__main__":
    main()
