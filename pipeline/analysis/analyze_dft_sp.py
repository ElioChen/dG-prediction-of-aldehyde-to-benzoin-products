#!/usr/bin/env python
"""Aggregate the 1% DFT single-point validation and characterise the
xTB-vs-DFT ΔG divergence.

Reads all chunk_*.csv produced by submit_dft_sp.sh, computes correlation /
error stats between ΔG(xTB) and ΔG(DFT r2SCAN-3c // xTB-RRHO), and -- the main
scientific question -- localises the large divergence seen in the negative xTB
tail by decomposing ΔΔG = ΔG_orca - ΔG_xtb into a reactant-side and a
product-side contribution.

One standalone figure per file (no composite panels). Output filenames are
timestamped so prior runs are never overwritten.
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EH2KCAL = 627.5094740631

RESULTS = "/gpfs/scratch1/shared/schen3/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c"


def load(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "chunk_*.csv")))
    if not files:
        raise SystemExit(f"no chunk_*.csv under {results_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"loaded {len(files)} chunks, {len(df)} rows")
    return df


def stats_block(x: np.ndarray, y: np.ndarray) -> dict:
    """y = reference (DFT), x = xTB."""
    resid = x - y
    return dict(
        n=len(x),
        mae=float(np.mean(np.abs(resid))),
        rmse=float(np.sqrt(np.mean(resid**2))),
        bias=float(np.mean(resid)),
        r=float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan"),
        max_abs=float(np.max(np.abs(resid))),
    )


NOISE_FLOOR = 3.2  # kcal/mol, see memory delta-mae-noise-floor


def gate_b(x: np.ndarray, y: np.ndarray) -> dict:
    """Decide whether full-library DFT-SP is justified.

    x = ΔG(xTB), y = ΔG(DFT). Returns metrics + a heuristic verdict. The user
    makes the final call; this just encodes the Gate B reasoning:
      - linear correctability: OLS y~x, residual MAE after correction. If that
        already reaches the ~3.2 kcal noise floor with high r, a cheap linear
        correction may obviate per-molecule DFT.
      - tail concentration: is the divergence concentrated in the most-negative
        xTB quintile (likely screen artifacts) rather than spread broadly?
    """
    # OLS linear fit y = a*x + b
    a, b = np.polyfit(x, y, 1)
    y_corr = a * x + b
    lin_resid = y - y_corr
    lin_mae = float(np.mean(np.abs(lin_resid)))
    lin_rmse = float(np.sqrt(np.mean(lin_resid**2)))

    raw_resid = x - y
    raw_mae = float(np.mean(np.abs(raw_resid)))

    # tail = most-negative xTB quintile
    q20 = np.quantile(x, 0.20)
    tail = x <= q20
    tail_mae = float(np.mean(np.abs(raw_resid[tail]))) if tail.sum() else float("nan")
    body_mae = float(np.mean(np.abs(raw_resid[~tail]))) if (~tail).sum() else float("nan")
    tail_ratio = tail_mae / body_mae if body_mae else float("nan")

    r = float(np.corrcoef(x, y)[0, 1])

    # verdict heuristic
    if lin_mae <= NOISE_FLOOR and r >= 0.90:
        verdict = ("LEAN-NO — xTB + a cheap linear correction already reaches the "
                   f"~{NOISE_FLOOR} kcal noise floor (lin-MAE={lin_mae:.2f}, r={r:.3f}). "
                   "Full per-molecule DFT-SP likely NOT worth ~25-32k core-h; "
                   "consider applying the linear correction instead, DFT only for the tail.")
    elif tail_ratio >= 2.5:
        verdict = ("CONDITIONAL — divergence is concentrated in the xTB negative tail "
                   f"(tail-MAE {tail_mae:.1f} vs body {body_mae:.1f}, {tail_ratio:.1f}×). "
                   "Likely screen artifacts/bad geometries. INSPECT Top-20 outlier "
                   "geometries first; clean/repair the tail before committing the full run.")
    elif lin_mae > NOISE_FLOOR:
        verdict = ("PASS — DFT changes ΔG broadly and is not removable by a linear "
                   f"correction (lin-MAE={lin_mae:.2f} > {NOISE_FLOOR}) nor confined to the "
                   "tail. Per-molecule DFT-SP adds real information → full run justified.")
    else:
        verdict = ("AMBIGUOUS — borderline; inspect figures and outliers manually.")

    return dict(a=a, b=b, lin_mae=lin_mae, lin_rmse=lin_rmse, raw_mae=raw_mae,
                r=r, tail_mae=tail_mae, body_mae=body_mae, tail_ratio=tail_ratio,
                q20=float(q20), verdict=verdict)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.results, "analysis")
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load(args.results)

    # ---- validity ----
    n_total = len(df)
    err = df["error"].notna() & (df["error"].astype(str).str.len() > 0)
    n_err = int(err.sum())
    ok = df["dG_xtb_kcal"].notna() & df["dG_orca_kcal"].notna() & (~err)
    d = df[ok].copy()
    x = d["dG_xtb_kcal"].to_numpy(float)
    y = d["dG_orca_kcal"].to_numpy(float)

    st = stats_block(x, y)
    gb = gate_b(x, y)

    d["abs_resid"] = np.abs(x - y)

    # NOTE: a reactant-vs-product split of ΔΔG from per-species G_orca − G_xtb is
    # ILL-POSED: GFN2-xTB (valence, E≈−60 Eh) and all-electron DFT (E≈−1200 Eh)
    # use incompatible absolute-energy references. The reference cancels ONLY in
    # the full reaction ΔG (benzoin condensation conserves atoms, benzoin = 2
    # aldehydes), so only ΔG_xtb / ΔG_orca are physically comparable. We instead
    # characterise the discrepancy by structural motif.
    smi = d["aldehyde_smiles"].astype(str)
    motifs = {
        "hypervalent-S  S(=O)(=O)": smi.str.contains(r"S(=O)(=O)", regex=False),
        "sulfonyl fluoride  S(=O)(=O)F": smi.str.contains("S(=O)(=O)F", regex=False),
        "triflate  OS(=O)(=O)C(F)(F)F": smi.str.contains("OS(=O)(=O)C(F)(F)F", regex=False),
        "nitro  [N+](=O)[O-]": smi.str.contains("[N+](=O)[O-]", regex=False),
    }

    # ---- figure 1: xTB vs DFT scatter ----
    fig, ax = plt.subplots(figsize=(6.2, 6))
    lim = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6, label="y = x")
    ax.scatter(x, y, s=14, alpha=0.55, edgecolor="none")
    ax.set_xlabel("ΔG(xTB GFN2/ALPB-DMSO)  [kcal/mol]")
    ax.set_ylabel("ΔG(r2SCAN-3c // xTB-RRHO, CPCM-DMSO)  [kcal/mol]")
    ax.set_title(f"1% DFT-SP validation  (n={st['n']})\n"
                 f"MAE={st['mae']:.2f}  RMSE={st['rmse']:.2f}  "
                 f"bias={st['bias']:+.2f}  r={st['r']:.3f}")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.legend(loc="upper left"); fig.tight_layout()
    f1 = os.path.join(outdir, f"scatter_xtb_vs_dft_{ts}.png")
    fig.savefig(f1, dpi=150); plt.close(fig)

    # ---- figure 2: residual vs xTB ΔG (exposes the tail) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="k", lw=1, alpha=0.5)
    ax.scatter(x, x - y, s=14, alpha=0.55, edgecolor="none")
    ax.set_xlabel("ΔG(xTB)  [kcal/mol]")
    ax.set_ylabel("ΔG(xTB) − ΔG(DFT)  [kcal/mol]")
    ax.set_title("xTB error vs xTB ΔG  (negative tail = screen artifacts?)")
    fig.tight_layout()
    f2 = os.path.join(outdir, f"residual_vs_xtb_{ts}.png")
    fig.savefig(f2, dpi=150); plt.close(fig)

    figs = [f1, f2]

    # ---- figure 3: decomposition (if components present) ----
    # ---- figure 3: residual distribution ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(x - y, bins=60, color="#3b6ea5", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(st["bias"], color="crimson", lw=1.5, ls="--",
               label=f"mean bias {st['bias']:+.1f}")
    ax.set_xlabel("ΔG(xTB) − ΔG(DFT)  [kcal/mol]")
    ax.set_ylabel("count")
    ax.set_title("xTB − DFT residual distribution")
    ax.legend(); fig.tight_layout()
    f3 = os.path.join(outdir, f"residual_hist_{ts}.png")
    fig.savefig(f3, dpi=150); plt.close(fig)
    figs.append(f3)

    # ---- merged CSV + outlier table ----
    merged_csv = os.path.join(outdir, f"dft_sp_merged_{ts}.csv")
    df.to_csv(merged_csv, index=False)

    cols = ["idx", "aldehyde_smiles", "dG_xtb_kcal", "dG_orca_kcal"]
    top = d.sort_values("abs_resid", ascending=False)[cols + ["abs_resid"]].head(20)

    def md_table(frame: pd.DataFrame) -> str:
        cs = list(frame.columns)
        out = ["| " + " | ".join(cs) + " |", "|" + "|".join(["---"] * len(cs)) + "|"]
        for _, r in frame.iterrows():
            cells = []
            for c in cs:
                v = r[c]
                cells.append(f"{v:+.2f}" if isinstance(v, (int, float, np.floating)) else str(v))
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    # ---- markdown report ----
    rep = os.path.join(outdir, f"REPORT_dft_sp_validation_{ts}.md")
    lines = []
    lines.append(f"# 1% DFT single-point ΔG validation — {ts}\n")
    lines.append(f"Source: `{args.results}` ({len(glob.glob(os.path.join(args.results, 'chunk_*.csv')))} chunks)\n")
    lines.append("## Coverage\n")
    lines.append(f"- rows total: **{n_total}** (target 2201)")
    lines.append(f"- with error flag: **{n_err}**")
    lines.append(f"- valid xTB+DFT pairs: **{st['n']}**\n")
    lines.append("## xTB vs DFT (r2SCAN-3c // xTB-RRHO)\n")
    lines.append("| metric | value (kcal/mol) |")
    lines.append("|---|---|")
    lines.append(f"| MAE | {st['mae']:.2f} |")
    lines.append(f"| RMSE | {st['rmse']:.2f} |")
    lines.append(f"| bias (xTB−DFT) | {st['bias']:+.2f} |")
    lines.append(f"| Pearson r | {st['r']:.3f} |")
    lines.append(f"| max \\|resid\\| | {st['max_abs']:.1f} |\n")
    lines.append(f"Mean bias **{st['bias']:+.2f}** kcal/mol = xTB systematically "
                 "predicts the benzoin condensation **too exergonic** vs DFT.\n")
    lines.append("## Where xTB fails — structural-motif enrichment\n")
    lines.append("_(A reactant-vs-product energy split is not computable: xTB and "
                 "all-electron DFT use incompatible absolute-energy references that "
                 "cancel only in the full reaction ΔG. So we attribute by motif.)_\n")
    hi = d["abs_resid"] >= d["abs_resid"].quantile(0.90)  # worst 10%
    lines.append("| motif | overall % | worst-10% % | mean \\|resid\\| with motif |")
    lines.append("|---|---|---|---|")
    for name, mask in motifs.items():
        ov = 100 * mask.mean()
        wr = 100 * mask[hi].mean()
        mr = float(d.loc[mask, "abs_resid"].mean()) if mask.any() else float("nan")
        lines.append(f"| {name} | {ov:.1f} | {wr:.1f} | {mr:.1f} |")
    lines.append("")
    # ---- Gate B decision section ----
    cov = st["n"] / 2201.0
    gate_a = "PASS" if (cov >= 0.95 and n_err / max(n_total, 1) < 0.05) else "CHECK"
    lines.append("## Gate decision — submit full library?\n")
    lines.append(f"**Gate A (clean run):** {gate_a} — coverage {cov*100:.1f}%, "
                 f"error rate {n_err/max(n_total,1)*100:.1f}%\n")
    lines.append("**Gate B (is DFT worth ~25-32k core-h?):**\n")
    lines.append(f"- linear fit ΔG_DFT = {gb['a']:.3f}·ΔG_xTB {gb['b']:+.2f}")
    lines.append(f"- residual MAE after linear correction: **{gb['lin_mae']:.2f}** "
                 f"(raw {gb['raw_mae']:.2f}); noise floor ≈ {NOISE_FLOOR}")
    lines.append(f"- tail concentration: tail-MAE {gb['tail_mae']:.1f} vs body "
                 f"{gb['body_mae']:.1f} ({gb['tail_ratio']:.1f}×, "
                 f"tail = xTB ΔG ≤ {gb['q20']:.1f})")
    lines.append(f"\n> **Verdict: {gb['verdict']}**\n")
    lines.append("_Heuristic — confirm by eyeballing the scatter/residual figures "
                 "and the outlier geometries below before committing the full run._\n")
    lines.append("**If submitting full library:** aromatic-only scope, add per-mol "
                 "`rmtree` to thermo_orca, `%128` throttle, `clean_orphan_scratch.sh` "
                 "between batches (see handoff memory).\n")
    lines.append("## Top-20 |residual| outliers\n")
    lines.append(md_table(top))
    lines.append("\n## Figures\n")
    for f in figs:
        lines.append(f"- `{os.path.basename(f)}`")
    with open(rep, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nwrote:\n  {rep}\n  {merged_csv}")
    for f in figs:
        print(f"  {f}")
    print(f"\nMAE={st['mae']:.2f}  RMSE={st['rmse']:.2f}  r={st['r']:.3f}  "
          f"n={st['n']}/{n_total}  err={n_err}")
    print(f"GATE B: {gb['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
