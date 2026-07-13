#!/usr/bin/env python3
"""
Ship the trained model into the package: copy the sweep's winning model + feature
spec into src/benzoin_dG/models/, and build the applicability-domain reference
(ad_reference.npz) from the exact training feature matrix.

Run after pipeline/sweep_delta.py (or train_delta.py) has written runs/models/.

  python pipeline/assemble_model.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

import delta_core as dc
import sys
sys.path.insert(0, str(dc.REPO_ROOT / "src"))
from benzoin_dG.applicability import build_reference  # noqa: E402

REPO = dc.REPO_ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=str(REPO / "runs/models"))
    ap.add_argument("--dest", default=str(REPO / "src/benzoin_dG/models"))
    ap.add_argument("--target", default=dc.DEFAULT_TARGET)
    ap.add_argument("--parquet", default=dc.DEFAULT_FEATURIZE_PARQUET,
                    help="training table used for the AD reference (match the sweep)")
    args = ap.parse_args()

    runs, dest = Path(args.runs), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # 1) copy model artifacts produced by the sweep/train
    for fn in ("delta_model.joblib", "feature_list.json", "metadata.json"):
        src = runs / fn
        if not src.exists():
            print(f"ERROR: {src} missing — run sweep_delta.py first")
            return 1
        shutil.copy2(src, dest / fn)
        print(f"copied {fn}")

    # 2) build AD reference from the SAME training table the model used
    tbl = dc.load_training_table(target=args.target, parquet=args.parquet)   # QC applied
    payload = build_reference(tbl.X.to_numpy())
    np.savez_compressed(dest / "ad_reference.npz", **payload)
    print(f"\nAD reference: {len(tbl.X)} training molecules, {tbl.X.shape[1]} features")
    print(f"  in-domain threshold  (p90 NN dist) = {payload['p_in']:.2f}")
    print(f"  extrapolation thresh (p99 NN dist) = {payload['p_edge']:.2f}")
    print(f"Wrote {dest/'ad_reference.npz'}")
    print(f"\nModel shipped to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
