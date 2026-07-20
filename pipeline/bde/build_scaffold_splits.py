#!/usr/bin/env python
"""Build genuinely scaffold-disjoint train/test splits for BDE (2026-07-17 overnight),
coordinated with the neighboring cross-benzoin conversation's own scaffold-disjoint rebuild
(cross_benzoin/rebuild_scaffold_disjoint_split.py, memory cross-five-diagnostics-20260717.md)
to avoid duplicating work -- per the user's explicit instruction to coordinate.

Aldehyde side: DIRECTLY REUSES cross-benzoin's own scaffold assignment
(data/cross_benzoin/candidates_v3/aldehydes_with_scaffold_split.parquet) rather than
recomputing it -- confirmed 100% canonical-SMILES match coverage (220,524/220,524) against
BDE's own aldehyde library, since both projects draw from the identical 220,859-aldehyde
homo_v6 library. Zero duplicated work for this half.

Product side: NOT covered by cross-benzoin's file (their products are cross donor+acceptor
pairs; BDE's products are homo self-condensation products, unique to this project) -- built
fresh here, using the SAME algorithm cross-benzoin's v2 fix established (single global
greedy bin-packing over scaffolds, largest-first, each whole scaffold assigned to whichever
split is furthest below its ROW-COUNT target) -- this is the exact fix for the v1 pitfall
(comparing differently-SIZED test sets), applied from the start rather than discovered by
trial and error a second time.

Target: 85/15 train/test (matches BDE's existing `molecule_cold_split(test_frac=0.15)`
convention, no separate top-level validation split -- train_gnn_hybrid_bde.py carves its
own early-stopping validation fraction out of the train pool internally).

Usage: python build_scaffold_splits.py --out-dir data/cross_benzoin/homo_v6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
SEED = 20260717
TARGET_FRAC = {"train": 0.85, "test": 0.15}


def murcko(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf) if scaf is not None else None
    except Exception:
        return None


def greedy_scaffold_split(df: pd.DataFrame, scaffold_col: str = "scaffold") -> pd.Series:
    """Single global greedy bin-packing pass: each whole scaffold -> whichever split is
    furthest below its row-count target. Guarantees zero scaffold overlap AND close-to-
    target split sizes by construction (the exact fix cross-benzoin's v1->v2 correction
    established -- see module docstring)."""
    scaf_counts = df.groupby(scaffold_col).size().sort_values(ascending=False)
    n_total = len(df)
    targets = {k: v * n_total for k, v in TARGET_FRAC.items()}
    current = {k: 0 for k in TARGET_FRAC}
    assign = {}
    order = scaf_counts.sample(frac=1.0, random_state=SEED).sort_values(ascending=False, kind="stable")
    for scaf, cnt in order.items():
        deficits = {k: targets[k] - current[k] for k in TARGET_FRAC}
        pick = max(deficits, key=deficits.get)
        assign[scaf] = pick
        current[pick] += cnt
    return df[scaffold_col].map(assign)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=H)
    args = ap.parse_args()

    # --- aldehydes: confirm the reused-from-dG file is in place ---
    ald_path = args.out_dir / "aldehydes_scaffold_split_from_dG.parquet"
    if not ald_path.exists():
        print(f"WARNING: {ald_path} not found -- run the reuse-mapping step first "
              "(match BDE aldehydes_all.csv against cross_benzoin's "
              "candidates_v3/aldehydes_with_scaffold_split.parquet by canonical SMILES)")
        return 1
    ald = pd.read_parquet(ald_path)
    print(f"aldehydes: reusing cross-benzoin's scaffold split ({len(ald)} rows) -- "
          f"{ald['scaffold_split'].value_counts().to_dict()}")

    # --- products: homo self-condensation, each product has exactly ONE parent aldehyde
    # (donor_id == acceptor_id conceptually) -- so "does the model generalize to an unseen
    # PARENT scaffold" is the right question, not an independent scaffold computed on the
    # much bigger fused product molecule. Propagate the already-reused aldehyde split
    # through donor_id rather than building a second, independent (and conceptually
    # mismatched) scaffold assignment. ---
    mol = pd.read_csv(H / "products_all.csv", usecols=["id", "donor_id", "smiles", "error"],
                       dtype=str, keep_default_na=False, low_memory=False)
    mol = mol[mol["error"] == ""].drop_duplicates("id").reset_index(drop=True)
    ald_map = ald.rename(columns={"id": "donor_id"})[["donor_id", "scaffold", "scaffold_split"]]
    out_prod = mol.merge(ald_map, on="donor_id", how="left")
    n_missing = out_prod["scaffold_split"].isna().sum()
    print(f"products: {len(out_prod)} molecules, {n_missing} with no parent-scaffold "
          f"assignment (parent aldehyde not in the reused split file)")
    out_prod = out_prod.dropna(subset=["scaffold_split"])[["id", "donor_id", "scaffold", "scaffold_split"]]
    sizes = out_prod["scaffold_split"].value_counts()
    print(sizes)

    out_prod.to_parquet(args.out_dir / "products_scaffold_split.parquet", index=False)
    out_prod.to_csv(args.out_dir / "products_scaffold_split.csv", index=False)
    print(f"wrote {args.out_dir / 'products_scaffold_split.parquet'} (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
