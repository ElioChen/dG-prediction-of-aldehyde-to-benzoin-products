"""CLI:  benzoin-dg "O=Cc1ccccc1"  [--json] [--n-confs N]"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass

from .predict import _format, predict_dG, predict_dG_champion, predict_dG_fast


def _as_dict(pred) -> dict:
    if hasattr(pred, "as_dict"):
        return pred.as_dict()
    if is_dataclass(pred):
        return asdict(pred)
    return dict(pred)


def _format_champion(pred) -> str:
    if pred.dG_pred is None:
        return f"{pred.smiles}\n  ERROR: {pred.error}"
    verdict = "FAVORABLE  (ΔG<0)" if pred.dG_pred < 0 else "unfavorable (ΔG>0)"
    route = "route-to-DFT" if pred.route_to_dft else "model-use"
    return (f"{pred.smiles}\n"
            f"  ΔG = {pred.dG_pred:+6.1f} kcal/mol   {verdict}\n"
            f"  uncertainty: {pred.uncertainty:.2f}   ({route})\n"
            f"  model: champion_ensemble72_20260626"
            f"   [g-xTB baseline {pred.dG_gxtb:+.1f}, Δ-correction {pred.dG_correction:+.1f}]")


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
    ap.add_argument("--champion", action="store_true",
                    help="use the full-library 72-feature champion path (slower; needs "
                         "xTB, cross_benzoin/pipeline code, and the bde_lite env)")
    ap.add_argument("--xtb-cores", type=int, default=2,
                    help="xTB cores for --champion")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    if args.fast and args.champion:
        print("error: choose only one of --fast or --champion", file=sys.stderr)
        return 2

    out = []
    for smi in args.smiles:
        try:
            if args.champion:
                p = predict_dG_champion(smi, xtb_bin=args.xtb_bin,
                                         n_confs=args.n_confs,
                                         xtb_cores=args.xtb_cores)
            elif args.fast:
                p = predict_dG_fast(smi, models_dir=args.models_dir)
            else:
                p = predict_dG(smi, xtb_bin=args.xtb_bin,
                               multiwfn_bin=args.multiwfn_bin, n_confs=args.n_confs,
                               models_dir=args.models_dir)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        out.append(_as_dict(p))
        if not args.json:
            print(_format_champion(p) if args.champion else _format(p))
    if args.json:
        json.dump(out if len(out) > 1 else out[0], sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
