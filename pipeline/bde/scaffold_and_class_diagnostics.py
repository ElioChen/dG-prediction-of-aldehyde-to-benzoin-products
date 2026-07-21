#!/usr/bin/env python
"""BDE architecture gap-check (2026-07-17), directly inspired by the neighboring cross-benzoin
conversation's "five diagnostics" (cross_benzoin/analysis/{cross_error_diagnostics,
scaffold_split_cv_cross,candidates_v3_representativeness}.py) -- their scaffold-split test
found the model's reported random-split R^2 (0.69-0.76) drops hard under a scaffold-disjoint
split (0.35-0.41), meaning candidates_v3's molecule-level splits likely leak structural
information the model can exploit. BDE (B4/B5/B6) has NEVER checked this -- `molecule_cold_split`
only guarantees the same molecule ID isn't in both train/test, not that train/test don't share
near-identical Bemis-Murcko scaffolds.

Three checks, all zero-new-compute (reuses B6's already-saved test predictions):
1. Scaffold-leakage check: for the aldehyde and product B6 test sets, how many TEST scaffolds
   also appear in TRAIN? If high, the reported R^2=0.90/0.92 may be optimistic the same way
   cross-benzoin's was.
2. Error by aliphatic/aromatic class (`aldehyde_class.parquet` cho_class, same source cross's
   own class_pair diagnostic used) -- does BDE show the same "aliphatic is hardest" pattern
   cross found?
3. Heteroscedasticity / relative-error check on the product-vs-aldehyde MAE gap (deeper dive
   on "product MAE ~2x aldehyde, mostly label-scale" from the earlier label-distribution
   analysis) -- does error scale with |y_true| (proportional/relative error roughly constant),
   or is there a genuine architecture-driven excess beyond what label scale explains?

Usage: python scaffold_and_class_diagnostics.py --outdir /tmp/bde_scaffold_diag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import molecule_cold_split  # noqa: E402

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
LOGS = Path("/scratch-shared/schen3/benzoin-dg/runs/logs")


def murcko(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf) if scaf is not None else None
    except Exception:
        return None


def analyze_one(which: str, split_col: str, outdir: Path, pred_prefix: str):
    pred = pd.read_csv(LOGS / f"{pred_prefix}_{which}_test_predictions.csv", dtype={"id": str})
    pred["err"] = (pred["y_pred"] - pred["y_true"]).abs()

    id_cols = ["id", "smiles", "error"] if which == "aldehydes" else ["id", "donor_id", "smiles", "error"]
    mol = pd.read_csv(H / f"{which}_all.csv", usecols=id_cols, dtype=str, keep_default_na=False, low_memory=False)
    mol = mol[mol["error"] == ""]
    labels = pd.read_csv(H / f"{which}_bdfe_gxtb_descriptors.csv", dtype={"id": str}).dropna(subset=["bde_gxtb_kcal"])
    from qc import qc_filter
    labels = labels[qc_filter(labels["bde_gxtb_kcal"])]
    df = labels.merge(mol, on="id", how="inner").dropna(subset=[]).reset_index(drop=True)
    df = df[df["smiles"] != ""].reset_index(drop=True)

    # reconstruct the SAME molecule_cold_split (seed=0, test_frac=0.15) the b6_best run used
    split = molecule_cold_split(df[split_col], test_frac=0.15, seed=0)
    df["_split"] = split
    df["scaffold"] = df["smiles"].map(murcko)

    train_scafs = set(df.loc[df["_split"] == "train", "scaffold"].dropna())
    test_df = df[df["_split"] == "test"].copy()
    test_df["scaffold_in_train"] = test_df["scaffold"].isin(train_scafs)
    leak_frac = test_df["scaffold_in_train"].mean()
    print(f"[{which}] scaffold leakage: {leak_frac:.1%} of test-set scaffolds also appear in train "
          f"(n_test={len(test_df)}, n_test_scaffolds={test_df['scaffold'].nunique()})", flush=True)

    merged = pred.merge(test_df[["id", "scaffold_in_train", "smiles"]], on="id", how="inner")
    by_leak = merged.groupby("scaffold_in_train")["err"].agg(["mean", "count"])
    print(f"[{which}] MAE by scaffold-in-train:\n{by_leak}\n", flush=True)

    # aliphatic/aromatic class (ids stored inconsistently -- "2" in aldehyde_class.parquet
    # vs "2.0" everywhere else in this script -- normalize both to a plain int-string)
    cls = pd.read_parquet(H / "aldehyde_class.parquet")
    cls["id"] = cls["id"].astype(int).astype(str)
    merged["_id_int"] = merged["id"].astype(float).astype(int).astype(str)
    if which == "products":
        merged = merged.merge(mol[["id", "donor_id"]], on="id", how="left")
        merged["_donor_int"] = merged["donor_id"].astype(float).astype(int).astype(str)
        merged = merged.merge(cls.rename(columns={"id": "_donor_int", "cls": "donor_cls"}),
                               on="_donor_int", how="left")
        by_cls = merged.groupby("donor_cls")["err"].agg(["mean", "count"])
    else:
        merged2 = merged.merge(cls.rename(columns={"id": "_id_int"}), on="_id_int", how="left")
        by_cls = merged2.groupby("cls")["err"].agg(["mean", "count"])
    print(f"[{which}] MAE by aliphatic/aromatic class:\n{by_cls}\n", flush=True)

    # heteroscedasticity: relative error vs |y_true|
    m2 = pred.merge(test_df[["id"]], on="id", how="inner") if which == "aldehydes" else pred
    m2 = pred.copy()
    m2["abs_y"] = m2["y_true"].abs()
    m2["rel_err"] = m2["err"] / m2["abs_y"].clip(lower=1.0)
    corr = np.corrcoef(m2["abs_y"], m2["err"])[0, 1]
    print(f"[{which}] corr(|y_true|, abs_err) = {corr:.3f}  mean_rel_err = {m2['rel_err'].mean():.4f}  "
          f"overall MAE = {pred['err'].mean():.3f}\n", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    test_df.merge(pred, on="id", how="inner")[
        ["id", "smiles", "scaffold", "scaffold_in_train", "y_true", "y_pred", "err"]
    ].to_csv(outdir / f"{which}_scaffold_diag.csv", index=False)
    return {"which": which, "leak_frac": leak_frac, "corr_abs_y_err": corr,
            "mae_leaked": float(by_leak.loc[True, "mean"]) if True in by_leak.index else None,
            "mae_novel": float(by_leak.loc[False, "mean"]) if False in by_leak.index else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--pred-prefix", default="bde_gnn_hybrid_tuned",
                     help="runs/logs/{prefix}_{which}_test_predictions.csv -- "
                          "'bde_gnn_hybrid_tuned' for B6, 'bondnet_bde' for B5")
    args = ap.parse_args()
    summary = []
    summary.append(analyze_one("aldehydes", "id", args.outdir, args.pred_prefix))
    summary.append(analyze_one("products", "donor_id", args.outdir, args.pred_prefix))
    pd.DataFrame(summary).to_csv(args.outdir / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
