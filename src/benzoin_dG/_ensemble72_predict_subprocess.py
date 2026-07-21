#!/usr/bin/env python
"""Standalone predictor for the ENSEMBLE72 bundle -- run with a numpy>=2
interpreter (e.g. envs/bde_lite), NOT the main nhc-workflow venv (numpy
1.26.4, which cannot unpickle this bundle's RandomState/BitGenerator objects
-- see `ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known
BitGenerator module`, same issue flagged in memory
`cross-benzoin-push-20260714`).

Reads one JSON object from stdin: {"features": {<72 FEATS keys>: value, ...}}
Writes one JSON object to stdout: {"correction": float, "uncertainty": float,
"route_to_dft": bool} or {"error": str}.

Kept as a tiny, dependency-free bridge script (only needs joblib/numpy on
whatever interpreter runs it) rather than merging numpy>=2 into the main
inference process, which would risk a numpy ABI clash with everything else
`_ensemble72_inference.py` already imports (RDKit, xTB wrappers) under the
project's numpy-1.26 venv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MODEL_PATH = (Path(__file__).resolve().parents[2]
              / "pipeline" / "models" / "gxtb_dft_correction_ENSEMBLE72_20260626.joblib")


def main() -> int:
    import numpy as np
    import joblib

    payload = json.loads(sys.stdin.read())
    feats_order = payload["feature_order"]
    row = payload["features"]

    bundle = joblib.load(MODEL_PATH)
    X = np.array([[row[k] for k in feats_order]], dtype=float)
    Xs = bundle["scaler"].transform(X)

    preds = [float(m.predict(Xs)[0]) for _, m in bundle["members"]]
    correction = float(np.mean(preds))
    q_lo = float(bundle["quantiles"][0.05].predict(Xs)[0])
    q_hi = float(bundle["quantiles"][0.95].predict(Xs)[0])
    width = q_hi - q_lo
    route = bool(width >= bundle["route_width_threshold"])

    print(json.dumps({"correction": correction, "uncertainty": width, "route_to_dft": route}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 -- surface any failure as JSON, not a traceback
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(1)
