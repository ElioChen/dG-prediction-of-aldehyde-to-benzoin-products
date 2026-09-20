#!/usr/bin/env python
"""Flying-dataset step 6 pilot, part 2: draw a uniform random sample from the
real 220,859^2 (~4.88e10) cross-pair space -- the space `candidates_v3` got
wrong -- and score it with the zero-new-compute lazy-absolute screener
(`train_lazy_absolute_screener.py`). Zero DFT/xTB spent: every feature comes
from the two aldehydes' existing caches + RDKit on the generated product
SMILES. Purpose: give the next acquisition-strategy decision (PROJECT_PLAN.md
sec2.7) real numbers about what the untouched 99.93%-of-space actually looks
like, instead of reasoning from the old (wrongly-constructed) candidates_v3
pool.

Usage:
    /home/schen3/venv/nequip/bin/python cross_benzoin/screen_flying_dataset_sample.py --n 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chemical_space import FlyingDataset  # noqa: E402

KNOWN_PAIRS = REPO / "data/chemical_space/labeled_pairs.parquet"
MODEL_DIR = REPO / "data/chemical_space/lazy_screen_model"
OUT = REPO / "data/chemical_space/flying_dataset_screen_pilot"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = joblib.load(MODEL_DIR / "model.joblib")
    feats = json.loads((MODEL_DIR / "feature_list.json").read_text())
    medians = pd.read_json(MODEL_DIR / "medians.json", typ="series")

    known = pd.read_parquet(KNOWN_PAIRS)
    fd = FlyingDataset(known_pairs=known)

    t0 = time.time()
    rows = fd.sample(args.n, seed=args.seed)
    print(f"sampled {len(rows)} never-labeled cross pairs in {time.time() - t0:.1f}s "
          f"(0 DFT/xTB calls -- purely cache + RDKit-on-SMILES)")

    lazy_df = pd.DataFrame([r["lazy_features"] for r in rows])
    X = lazy_df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").fillna(medians)
    pred = model.predict(X)

    out_df = pd.DataFrame({
        "donor_idx": [r["donor_idx"] for r in rows],
        "acceptor_idx": [r["acceptor_idx"] for r in rows],
        "reaction_type": [r["reaction_type"] for r in rows],
        "product_smiles": [r["product_smiles"] for r in rows],
        "pred_dG_r2scan_kcal_lazy": pred,
    })

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"sample_n{args.n}_seed{args.seed}.parquet"
    out_df.to_parquet(out_path)

    known_labels = known["label"].dropna().to_numpy()
    summary = {
        "built_by": "cross_benzoin/screen_flying_dataset_sample.py",
        "n_sampled": len(out_df),
        "dft_or_xtb_calls_spent": 0,
        "pred_dG_stats": {
            "mean": float(np.mean(pred)), "std": float(np.std(pred)),
            "p10": float(np.percentile(pred, 10)), "median": float(np.median(pred)),
            "p90": float(np.percentile(pred, 90)),
            "frac_favorable_lt0": float(np.mean(pred < 0)),
        },
        "known_labeled_pairs_dG_stats_for_comparison": {
            "n": len(known_labels), "mean": float(np.mean(known_labels)),
            "std": float(np.std(known_labels)), "median": float(np.median(known_labels)),
            "frac_favorable_lt0": float(np.mean(known_labels < 0)),
        },
        "reaction_type_counts": out_df["reaction_type"].value_counts(dropna=False).to_dict(),
        "note": "pred is from the lazy-only ABSOLUTE screener (holdout MAE 3.039, R2 0.315 "
                "on labeled pairs) -- a first-pass ranking signal only, not a substitute for "
                "the deployed champion (needs real product geometry it doesn't have here).",
    }
    (OUT / f"summary_n{args.n}_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
