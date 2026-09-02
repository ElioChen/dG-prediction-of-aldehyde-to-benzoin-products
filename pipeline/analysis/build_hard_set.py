#!/usr/bin/env python
"""Assemble the project-wide "hard set": the molecules our models predict worst, tagged by
*why* they are hard, across every task (homo BDE aldehyde / homo BDE product / homo dG /
cross dG) and both sides (reactant / product).

This is deliberately DYNAMIC. It reads whatever the latest per-task prediction CSVs are,
re-derives residuals + cause tags, writes `hard_set.parquet` + `hard_set_summary.json`, and
snapshots both under `hard_set_history/<UTC timestamp>/`. Re-run it after any model retrain;
compare snapshots to see which molecules we have "overcome" (dropped out of the set) and
which are stubborn. The Streamlit app `dashboard/hard_set_app.py` reads `hard_set.parquet`.

Cause tags (a molecule can carry several):
  gxtb_baseline_failure   |g-xTB baseline − DFT| is itself huge (Δ-learning can't recover a
                          broken baseline) -- see
                          pipeline/analysis/notes/gxtb_baseline_failure_hardtail_20260902.md
                          and pipeline/bde/notes/nitro_delta_sp_bimodal_20260902.md
  fg:<group>              carries a known-hard substructure (P, P=O, phosphonium, sulfonyl,
                          nitro, imine, amide, N-oxide, triflate, boron, selenium, ...)
  heavy_heteroatom        contains Se / Te / As / heavy halogen
  flexible                many rotatable bonds / large -- single-conformer label noisier
  high_uncertainty        model's own uncertainty flag / PI width is in the top decile
  (scaffold_novel)        only tagged when the source provides a scaffold_split column

Usage:
  python pipeline/analysis/build_hard_set.py            # rebuild from latest predictions
  python pipeline/analysis/build_hard_set.py --top-frac 0.15 --min-abs-res 3.0
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]
VIZ = REPO / "data/cross_benzoin/homo_v6/viz_gxtb_20260625"
OUT_DIR = REPO / "data/analysis/hard_set"
HARD_SET = OUT_DIR / "hard_set.parquet"
SUMMARY = OUT_DIR / "hard_set_summary.json"
HISTORY = OUT_DIR / "hard_set_history"

# ---------------------------------------------------------------------------
# Sources. Each is (task, side, builder). A builder returns a DataFrame with at least:
#   id, smiles, label, pred, baseline (may be NaN), [uncertainty], [scaffold_split]
# Missing files are skipped with a warning so the set is whatever data currently exists.
# ---------------------------------------------------------------------------

def _lib_smiles_by_index() -> pd.Series:
    lib = pd.read_csv(REPO / "data/library/aldehydes_clean_v6.csv", usecols=["SMILES"])
    return pd.Series(lib["SMILES"].values, index=np.arange(len(lib)))


def src_homo_dg() -> pd.DataFrame | None:
    f = VIZ / "test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv"
    if not f.exists():
        print(f"  skip homo_dg: {f.name} missing")
        return None
    d = pd.read_csv(f)
    out = pd.DataFrame({
        "id": d["id"].astype(str),
        "smiles": d["smiles"],            # product SMILES
        "ald_smiles": d.get("ald_smiles"),
        "label": d["dG_orca_kcal"],
        "pred": d["dG_pred"],
        "baseline": d["dG_gxtb_kcal"],
        "uncertainty": d.get("uncertainty_pi_width"),
        "cls": d.get("cls"),
    })
    return out


def src_cross_dg() -> pd.DataFrame | None:
    # prefer the recovered rounds-1-7 ensemble CV predictions (job train_r17, 2026-09-02)
    cands = [
        REPO / "data/cross_benzoin/cross_round7/train_ensemble_7rounds_recovered_v1/data/cv_predictions.csv",
        REPO / "data/cross_benzoin/cross_round7/train_ensemble_7rounds_slim120_v1/data/cv_predictions.csv",
    ]
    f = next((c for c in cands if c.exists()), None)
    if f is None:
        print("  skip cross_dg: no cv_predictions.csv found")
        return None
    d = pd.read_csv(f)
    lib = _lib_smiles_by_index()
    def smi(idx):
        try:
            return lib.get(int(float(idx)))
        except (ValueError, TypeError):
            return None
    out = pd.DataFrame({
        "id": d["id"].astype(str),
        "smiles": d["donor_id"].map(smi),          # donor aldehyde (reactant side)
        "ald_smiles": d["acceptor_id"].map(smi),   # acceptor aldehyde
        "label": d["dG_orca_kcal"],
        "pred": d["dG_pred"],
        "baseline": d["dG_gxtb_kcal"],
        "uncertainty": np.nan,
        "cls": d.get("reaction_type"),
    })
    return out


def src_bde(which: str) -> pd.DataFrame | None:
    # populated once the post-purge B6 scaffold-disjoint retrain lands its *_pred.csv
    f = REPO / f"runs/logs/scaffold_disjoint_bde/{which}_scaffold_disjoint_ckpt_pred.csv"
    if not f.exists():
        f = REPO / f"runs/logs/scaffold_disjoint_bde/{which}_scaffold_disjoint_pred.csv"
    if not f.exists():
        print(f"  skip bde_{which}: no *_pred.csv yet (B6 retrain pending)")
        return None
    d = pd.read_csv(f)
    lc = next((c for c in ("y_true", "bde_gxtb_kcal", "label", "true") if c in d.columns), None)
    pc = next((c for c in ("y_pred", "pred", "prediction") if c in d.columns), None)
    sc = next((c for c in ("smiles", "SMILES") if c in d.columns), None)
    if not (lc and pc and sc):
        print(f"  skip bde_{which}: unexpected columns {list(d.columns)[:8]}")
        return None
    return pd.DataFrame({
        "id": d.get("id", pd.RangeIndex(len(d))).astype(str),
        "smiles": d[sc], "ald_smiles": None,
        "label": d[lc], "pred": d[pc], "baseline": np.nan, "uncertainty": np.nan,
        "cls": None,
    })


SOURCES = [
    ("homo_dg", "product", src_homo_dg),
    ("cross_dg", "reactant", src_cross_dg),
    ("bde_aldehyde", "reactant", lambda: src_bde("aldehydes")),
    ("bde_product", "product", lambda: src_bde("products")),
]

# ---------------------------------------------------------------------------
HARD_SMARTS = {
    "fg:phosphine_oxide": "[#15]=O",
    "fg:phosphonium": "[#15+]",
    "fg:phosphorus": "[#15]",
    "fg:sulfonyl": "[#16X4](=[OX1])(=[OX1])",
    "fg:sulfinyl": "[#16X3](=[OX1])",
    "fg:nitro": "[NX3+](=O)[O-]",
    "fg:N_oxide": "[#7+][O-]",
    "fg:imine": "[CX3]=[NX2]",
    "fg:amide": "[CX3](=O)[NX3]",
    "fg:triflate": "[#16](=O)(=O)OC(F)(F)F",
    "fg:boron": "[#5]",
    "fg:azide": "[NX2]=[NX2+]=[NX1-]",
    "fg:diazo": "[CX3]=[NX2+]=[NX1-]",
}
HARD_PATS = {k: Chem.MolFromSmarts(v) for k, v in HARD_SMARTS.items()}
HEAVY = {"Se", "Te", "As", "Br", "I"}
BASELINE_FAIL_KCAL = 15.0
FLEXIBLE_ROTB = 12
FLEXIBLE_MW = 500.0


def tag_molecule(smiles_list) -> tuple[set, dict]:
    tags: set[str] = set()
    props = {"mw": np.nan, "rotb": np.nan, "n_heavy_atoms": np.nan}
    for smi in smiles_list:
        if not isinstance(smi, str) or not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for k, pat in HARD_PATS.items():
            if pat is not None and m.HasSubstructMatch(pat):
                tags.add(k)
        syms = {a.GetSymbol() for a in m.GetAtoms()}
        if syms & HEAVY:
            tags.add("heavy_heteroatom")
        if rdMolDescriptors.CalcNumAromaticRings(m) == 0:
            tags.add("aliphatic")
        mw = Descriptors.MolWt(m)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(m)
        props["mw"] = np.nanmax([props["mw"], mw]) if not np.isnan(props["mw"]) else mw
        props["rotb"] = np.nanmax([props["rotb"], rotb]) if not np.isnan(props["rotb"]) else rotb
        props["n_heavy_atoms"] = m.GetNumHeavyAtoms()
    if props["rotb"] >= FLEXIBLE_ROTB or (props["mw"] or 0) >= FLEXIBLE_MW:
        tags.add("flexible")
    # collapse the phosphorus ladder to the most specific
    if "fg:phosphine_oxide" in tags or "fg:phosphonium" in tags:
        tags.discard("fg:phosphorus")
    return tags, props


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-frac", type=float, default=0.15,
                    help="per-source: keep this fraction with the largest |residual|")
    ap.add_argument("--min-abs-res", type=float, default=None,
                    help="also/instead keep every row with |residual| >= this (kcal/mol)")
    ap.add_argument("--no-history", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for task, side, builder in SOURCES:
        print(f"[{task}]")
        df = builder()
        if df is None or not len(df):
            continue
        df = df.dropna(subset=["label", "pred"]).copy()
        df["residual"] = df["pred"] - df["label"]
        df["abs_residual"] = df["residual"].abs()
        df["baseline_err"] = df["baseline"] - df["label"]
        df["pctile_abs_res"] = df["abs_residual"].rank(pct=True)
        thr = df["abs_residual"].quantile(1 - args.top_frac)
        keep = df["abs_residual"] >= thr
        if args.min_abs_res is not None:
            keep |= df["abs_residual"] >= args.min_abs_res
        df = df[keep].copy()
        df["task"], df["default_side"] = task, side

        causes, mw, rotb, nha = [], [], [], []
        for _, r in df.iterrows():
            tg, pr = tag_molecule([r.get("smiles"), r.get("ald_smiles")])
            if pd.notna(r["baseline_err"]) and abs(r["baseline_err"]) >= BASELINE_FAIL_KCAL:
                tg.add("gxtb_baseline_failure")
            # benzoin dG lives in ~0-15 kcal; BDE in ~30-120. A label/baseline far outside
            # its task's physical window is a data-quality outlier, not model difficulty.
            phys_hi = 130.0 if str(task).startswith("bde") else 60.0
            if abs(r["label"]) > phys_hi:
                tg.add("implausible_label")
            if pd.notna(r["baseline"]) and abs(r["baseline"]) > phys_hi:
                tg.add("implausible_baseline")
            if pd.notna(r.get("uncertainty")):
                tg.add("_has_unc")
            causes.append(sorted(t for t in tg if not t.startswith("_")))
            mw.append(pr["mw"]); rotb.append(pr["rotb"]); nha.append(pr["n_heavy_atoms"])
        df["causes"] = causes
        df["mw"], df["rotb"], df["n_heavy_atoms"] = mw, rotb, nha
        # high-uncertainty tag: top decile of the source's own uncertainty column
        if df["uncertainty"].notna().any():
            u_thr = df["uncertainty"].quantile(0.9)
            df["causes"] = [
                sorted(set(c) | ({"high_uncertainty"} if pd.notna(u) and u >= u_thr else set()))
                for c, u in zip(df["causes"], df["uncertainty"])
            ]
        df["causes"] = df["causes"].map(lambda c: c or ["unexplained"])
        df["n_causes"] = df["causes"].map(len)
        frames.append(df)
        print(f"  kept {len(df)} hard rows (|res| >= {thr:.2f} kcal, "
              f"worst {df['abs_residual'].max():.1f})")

    if not frames:
        print("no sources available -- nothing written")
        return 1
    hs = pd.concat(frames, ignore_index=True)
    cols = ["task", "default_side", "id", "smiles", "ald_smiles", "cls", "label", "pred",
            "residual", "abs_residual", "pctile_abs_res", "baseline", "baseline_err",
            "uncertainty", "mw", "rotb", "n_heavy_atoms", "n_causes", "causes"]
    hs = hs[[c for c in cols if c in hs.columns]]
    hs["causes"] = hs["causes"].map(list)
    hs["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    hs.to_parquet(HARD_SET, index=False)

    # explode causes for the summary
    ex = hs.explode("causes")
    summary = {
        "generated_at": hs["generated_at"].iloc[0],
        "n_total": int(len(hs)),
        "by_task": {t: int(n) for t, n in hs["task"].value_counts().items()},
        "by_cause": {c: int(n) for c, n in ex["causes"].value_counts().items()},
        "worst_per_task": {
            t: {"id": str(g.loc[g["abs_residual"].idxmax(), "id"]),
                "abs_residual": float(g["abs_residual"].max())}
            for t, g in hs.groupby("task")
        },
        "mae_of_hard_set_by_task": {
            t: float(g["abs_residual"].mean()) for t, g in hs.groupby("task")
        },
        "params": {"top_frac": args.top_frac, "min_abs_res": args.min_abs_res},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))

    if not args.no_history:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = HISTORY / ts
        snap.mkdir(parents=True, exist_ok=True)
        hs.to_parquet(snap / "hard_set.parquet", index=False)
        (snap / "hard_set_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"snapshot -> {snap.relative_to(REPO)}")

    print(f"\nwrote {HARD_SET.relative_to(REPO)}  ({len(hs)} rows)")
    print(json.dumps(summary["by_cause"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
