#!/usr/bin/env python3
"""
Fidelity check for the recomputed aldehyde descriptor library.

`aldehydes_all.csv` is the one lost file with no archive, so its xTB/descriptor
block had to be recomputed from scratch rather than restored. Before any rebuilt
training table is trusted, the recomputed numbers must be shown to agree with the
originals -- otherwise every downstream MAE is uninterpretable.

The reference is `cross_round3/cross_train_table_3rounds.csv`, which is tracked in
git and embeds the ORIGINAL `donor_*`/`acceptor_*` ALDEHYDE_FEATS values, joined
back then from the now-destroyed `aldehydes_all.csv`. Any aldehyde appearing both
there and in the recompute gives a direct old-vs-new comparison.

Note this is not expected to be bit-exact the way the archive restores were: xTB
conformer search is stochastic, so agreement is judged per-feature by relative
median deviation and correlation, not by equality.

Usage:
    python cross_benzoin/check_aldehyde_recompute_fidelity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cross_benzoin"))
from assemble_cross_training_table import ALDEHYDE_FEATS  # noqa: E402

NEW = REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv"
REF = REPO / "data/cross_benzoin/cross_round3/cross_train_table_3rounds.csv"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def main() -> int:
    new = pd.read_csv(NEW, low_memory=False)
    new["_c"] = new["smiles"].map(canon)
    new = new.dropna(subset=["_c"]).drop_duplicates("_c").set_index("_c")
    ref = pd.read_csv(REF, low_memory=False)

    # stack donor and acceptor sides into one old-vs-new comparison
    feats = [f for f in ALDEHYDE_FEATS if f != "bde_gxtb_kcal" and f in new.columns]
    rows = []
    for side in ("donor", "acceptor"):
        cols = [f"{side}_{f}" for f in feats]
        have = [c for c in cols if c in ref.columns]
        if len(have) != len(cols):
            print(f"WARN: reference missing {len(cols)-len(have)} {side} columns")
        sub = ref[[f"{side}_smiles"] + have].copy()
        sub["_c"] = sub[f"{side}_smiles"].map(canon)
        sub = sub[sub["_c"].isin(new.index)]
        sub = sub.rename(columns={f"{side}_{f}": f for f in feats})
        rows.append(sub[["_c"] + [f for f in feats if f in sub.columns]])
    old = pd.concat(rows, ignore_index=True).drop_duplicates("_c")
    if old.empty:
        print("no overlap between the recompute and the reference table yet")
        return 1
    print(f"overlapping aldehydes: {len(old)}\n")

    got = new.loc[old["_c"]]
    print(f"{'feature':26s} {'n':>6s} {'rel_med_dev':>12s} {'pearson_r':>10s} {'max_reldev':>11s}")
    print("-" * 70)
    bad = []
    for f in feats:
        if f not in old.columns:
            continue
        a = pd.to_numeric(old[f], errors="coerce").to_numpy()
        b = pd.to_numeric(got[f], errors="coerce").to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            continue
        scale = max(np.median(np.abs(a[m])), 1e-9)
        rel = np.abs(a[m] - b[m]) / scale
        r = np.corrcoef(a[m], b[m])[0, 1] if np.std(a[m]) > 0 and np.std(b[m]) > 0 else np.nan
        print(f"{f:26s} {m.sum():6d} {np.median(rel):12.5f} {r:10.4f} {np.max(rel):11.4f}")
        if np.median(rel) > 0.02 or (np.isfinite(r) and r < 0.98):
            bad.append(f)
    print()
    if bad:
        print(f"FEATURES OUT OF TOLERANCE ({len(bad)}): {bad}")
        print("tolerance = median relative deviation <= 2% and pearson r >= 0.98")
    else:
        print("all compared features within tolerance "
              "(median relative deviation <= 2%, pearson r >= 0.98)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
