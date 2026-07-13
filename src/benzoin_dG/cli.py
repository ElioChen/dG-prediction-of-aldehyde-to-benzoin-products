"""CLI:  benzoin-dg "O=Cc1ccccc1"  [--json] [--n-confs N]"""
from __future__ import annotations

import argparse
import json
import sys

from .predict import _format, predict_dG, predict_dG_fast


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="benzoin-dg",
        description="Predict benzoin-condensation ΔG (kcal/mol, r2SCAN-3c) from an "
                    "aldehyde SMILES. Reports the sign (favorable ΔG<0 vs "
                    "unfavorable ΔG>0), magnitude, and a confidence tier.")
    ap.add_argument("smiles", nargs="+", help="one or more aldehyde SMILES")
    ap.add_argument("--n-confs", type=int, default=5, help="conformers for featurization")
    ap.add_argument("--xtb-bin", default=None)
    ap.add_argument("--multiwfn-bin", default=None)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="use the pure-SMILES 2D surrogate (no xTB; instant, ~1 kcal "
                         "less accurate) instead of the xTB + Δ-correction model")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    out = []
    for smi in args.smiles:
        try:
            if args.fast:
                p = predict_dG_fast(smi, models_dir=args.models_dir)
            else:
                p = predict_dG(smi, xtb_bin=args.xtb_bin,
                               multiwfn_bin=args.multiwfn_bin, n_confs=args.n_confs,
                               models_dir=args.models_dir)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        out.append(p.as_dict())
        if not args.json:
            print(_format(p))
    if args.json:
        json.dump(out if len(out) > 1 else out[0], sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
