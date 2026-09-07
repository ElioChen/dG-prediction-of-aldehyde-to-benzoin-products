#!/usr/bin/env python3
"""GREEN/AMBER-branch prep: build the pair list for the B97-3c intermediate validation
(B97-3c dG on production funnel_v3 geometry for the frozen scaffold-disjoint holdout +
a stratified train slice, enough to retrain the Delta-model and measure whether the
champion holdout MAE actually drops below 2.215).

Feeds cross_benzoin/rec1_prodgeom_recheck_worker.py (same schema:
pid,grp,donor_smiles,acceptor_smiles,prod_smiles,label,gxtb_stored).

  python select_rec1_b973c_intermediate.py --n-train 1552 --seed 20260908
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
TAB = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet"
OUTDIR = REPO / "data/cross_benzoin/rec1_prodgeom_recheck"
COLS = ["id", "donor_smiles", "acceptor_smiles", "smiles", "round", "new_scaffold_split",
        "dG_orca_kcal", "dG_gxtb_kcal"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1552, help="train rows to add to the 448 holdout")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--max-heavy", type=int, default=110)
    a = ap.parse_args()

    df = pd.read_parquet(TAB, columns=COLS)
    df = df[df["dG_orca_kcal"].notna()].copy()
    # heavy-atom cap via product smiles length proxy is unreliable; use rdkit
    from rdkit import Chem
    df["prod_heavy"] = [
        sum(1 for at in m.GetAtoms() if at.GetSymbol() != "H") if (m := Chem.MolFromSmiles(str(s))) else 10**9
        for s in df["smiles"]
    ]
    df = df[df["prod_heavy"] <= a.max_heavy]

    hold = df[df["new_scaffold_split"] == "test"].copy()
    hold["grp"] = "holdout"
    trainable = df[df["new_scaffold_split"].isin(["train", "mixed"])].copy()

    # stratify the train slice by round x dG tertile
    trainable["dg_bin"] = pd.qcut(trainable["dG_orca_kcal"], 3, labels=False, duplicates="drop")
    frac = min(1.0, a.n_train / len(trainable))
    tr = (trainable.groupby(["round", "dg_bin"], group_keys=False)
          .sample(frac=frac, random_state=a.seed))
    if len(tr) > a.n_train:
        tr = tr.sample(a.n_train, random_state=a.seed)
    tr = tr.copy()
    tr["grp"] = "train_" + tr["round"].astype(str)

    out = pd.concat([hold, tr], ignore_index=True)
    out = out.rename(columns={"id": "pid", "smiles": "prod_smiles",
                              "dG_orca_kcal": "label", "dG_gxtb_kcal": "gxtb_stored"})
    keep = ["pid", "grp", "donor_smiles", "acceptor_smiles", "prod_smiles", "label",
            "gxtb_stored", "round", "new_scaffold_split", "prod_heavy"]
    out = out[keep]
    dst = OUTDIR / f"rec1_b973c_intermediate_{len(out)}.csv"
    out.to_csv(dst, index=False)
    print(f"wrote {len(out)} pairs ({len(hold)} holdout + {len(tr)} train) -> {dst}")
    print(out.groupby("grp").size().to_string())
    print(f"\nprod_heavy: min {out.prod_heavy.min()} p50 {int(out.prod_heavy.median())} "
          f"p90 {int(out.prod_heavy.quantile(0.9))} max {out.prod_heavy.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
