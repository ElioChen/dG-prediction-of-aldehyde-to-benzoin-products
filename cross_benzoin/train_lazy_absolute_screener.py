#!/usr/bin/env python
"""Flying-dataset step 6, zero-new-compute pilot (2026-09-20, user: "先不启动
[cross AL], 只把基础设施打通"): a screening model that needs NOTHING beyond
what FlyingDataset.pair() already returns for an arbitrary, never-labeled
cross pair -- i.e. trained on the LAZY feature tier only (no product QM /
product Mordred, which need a real geometry), and predicting the ABSOLUTE
label directly rather than a Delta against a baseline.

This is deliberately a different model from `ablation_lazy_vs_full_features.py`
(2026-09-11), which also restricted to the lazy tier but still Delta-learned
against `dG_b973c_kcal` -- fine for measuring "how much does the product-side
tier carry" on ALREADY-LABELED pairs (where the baseline is free, precomputed
in the table), but useless for actually screening a brand-new candidate pair,
where a baseline value doesn't exist without running B97-3c on that pair's
product geometry first (real, if cheap, compute -- exactly what "no new
compute" rules out here). Absolute-target is the only honest formulation for
a genuinely-zero-compute screen.

Also restricted to the ~99-feature intersection of the champion's lazy_feats
(166, per CHEMICAL_SPACE.md sec5b) and what FlyingDataset.pair()'s
lazy_features dict actually populates today (pair() doesn't yet join aldehyde
Mordred -- a real, fixable gap, flagged not fixed this pass; see module
docstring note below).

Usage:
    /home/schen3/venv/nequip/bin/python cross_benzoin/train_lazy_absolute_screener.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_core as dc  # noqa: E402
from train_cross_delta import _feature_blocks  # noqa: E402
from assemble_cross_training_table import PRODUCT_FEATS  # noqa: E402
from chemical_space import FlyingDataset  # noqa: E402

TABLE = REPO / "data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet"
TARGET_COL = "dG_r2scan_kcal"  # this table's own true label (b973c generation); train_cross_delta.TARGET_COL is the older g-xTB-era name and isn't a column here
KNOWN_PAIRS = REPO / "data/chemical_space/labeled_pairs.parquet"
OUTDIR = REPO / "data/chemical_space/lazy_screen_model"


def main() -> int:
    df = pd.read_parquet(TABLE)
    train_df = df[df["new_scaffold_split"] == "train"].reset_index(drop=True)
    test_df = df[df["new_scaffold_split"] == "test"].reset_index(drop=True)

    all_feats = [c for c in df.columns if c not in {
        "id", "donor_id", "acceptor_id", "pair_key", "reaction_type", "round",
        "donor_smiles", "acceptor_smiles", "smiles", "dG_xtb_kcal", "dG_gxtb_kcal",
        "dG_orca_kcal", "donor_scaf_split", "acceptor_scaf_split", "new_scaffold_split",
        "dG_r2scan_kcal", "dG_b973c_kcal", "dG_gxtb_kcal_selfconsistent",
        "dG_orca_kcal_stored", "label", "label_stored", "gxtb_stored", "grp"}]
    full_feats = _feature_blocks(all_feats)["all_raw_blocks+mordred"]
    not_lazy = set(PRODUCT_FEATS) | {c for c in full_feats if c.startswith("product_mordred_")}
    lazy_feats = [c for c in full_feats if c not in not_lazy]

    # what FlyingDataset.pair() can actually populate for a NEVER-labeled pair
    # -- see module docstring: aldehyde Mordred is not joined into pair()'s
    # lazy tier yet, so those ~67 columns are dropped here too, honestly.
    known = pd.read_parquet(KNOWN_PAIRS)
    fd = FlyingDataset(known_pairs=known)
    probe = fd.pair(0, 1)
    pair_lazy_keys = set(probe["lazy_features"].keys())
    usable_feats = [c for c in lazy_feats if c in pair_lazy_keys]
    dropped = [c for c in lazy_feats if c not in pair_lazy_keys]
    print(f"champion lazy_feats: {len(lazy_feats)}; usable via pair() today: {len(usable_feats)}; "
          f"dropped (not in pair()'s lazy tier, mostly aldehyde Mordred): {len(dropped)}")

    medians = train_df[usable_feats].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    Xtr = train_df[usable_feats].apply(pd.to_numeric, errors="coerce").fillna(medians)
    Xte = test_df[usable_feats].apply(pd.to_numeric, errors="coerce").fillna(medians)
    ytr = train_df[TARGET_COL].to_numpy()
    yte = test_df[TARGET_COL].to_numpy()

    model = dc.build_model("xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, 0)
    model.fit(Xtr, ytr)
    pred_te = model.predict(Xte)
    holdout = dc.metrics_vs_dft(yte, pred_te)
    print(f"[lazy-only, ABSOLUTE target, {len(usable_feats)} feats] "
          f"holdout MAE={holdout['MAE']:.3f}  R2={holdout['R2']:.3f}  "
          f"(for reference: champion Delta-model 0.528, g-xTB-no-model baseline 5.037, "
          f"09-11 lazy+Delta-on-known-pairs ablation 3.548 -- this number is expected to sit "
          f"worse than that, since a truly-new pair also lacks the Delta baseline the 09-11 "
          f"ablation still got for free)")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTDIR / "model.joblib")
    (OUTDIR / "feature_list.json").write_text(json.dumps(usable_feats, indent=2))
    medians.to_json(OUTDIR / "medians.json")
    (OUTDIR / "metadata.json").write_text(json.dumps({
        "built_by": "cross_benzoin/train_lazy_absolute_screener.py",
        "target": "absolute dG_r2scan_kcal (no Delta baseline -- none available for new pairs)",
        "n_feats": len(usable_feats),
        "n_feats_dropped_no_mordred": len(dropped),
        "n_train": len(train_df), "n_test": len(test_df),
        "holdout_MAE": holdout["MAE"], "holdout_R2": holdout["R2"],
        "known_pairs_excluded_from_screening": len(known),
    }, indent=2))
    print(f"saved model + feature_list + medians + metadata to {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
