#!/usr/bin/env python3
"""
Deep SMILES filter for benzoin aldehyde candidates.

Keeps only molecules that are valid substrates for the benzoin condensation and
clean enough for the xTB/DFT descriptor + ΔG pipeline. Rejects (with reasons):

  invalid_parse        RDKit cannot parse the SMILES
  multi_component      contains '.' (salt / mixture / counter-ion)
  charged              net formal charge != 0 (e.g. oxonium, free ions)
  disallowed_element   contains an element outside {H,C,N,O,F,S,Cl,Br,I}
  isotope              any isotope-labelled atom ([2H],[13C],...) — redundant for
                       ΔG/descriptors and breaks the benzoin builder
  zwitterion_or_nitro  any atom with a non-zero formal charge while net = 0:
                       nitro [N+](=O)[O-], N-oxide, nitronate, ylides, ...
  reactive_group       a reactive / xTB-unreliable functional group that is not a
                       realistic benzoin substrate (nitroso N=O, azo/diazo N=N,
                       hypervalent S–F, ketene/cumulene, isocyanide)
  not_single_aldehyde  number of TRUE aldehydes [CX3H1](=O)[#6] != 1
                       (excludes formic acid/ester/amide, acyl halide, formaldehyde)

The isotope / zwitterion / reactive-group rules were added after the n=1500
energy diagnostic (pipeline/analyze_energies.py): those classes carried 2–3× the
xTB→DFT correction noise (nitroso σ≈23, nitro σ≈15, azo σ≈13 vs the ~7 baseline)
and unphysical xTB geometries (ΔG to −160 kcal/mol).

Usage
  python ml/filter_smiles.py                                  # defaults
  python ml/filter_smiles.py --input aldehyde/aldehydes_benzoin.csv \
                             --output ml/data/aldehydes_clean.csv
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ALLOWED_Z = {1, 6, 7, 8, 9, 16, 17, 35, 53}   # H C N O F S Cl Br I
TRUE_ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=O)[#6]")  # R-CHO, R = carbon

# Reactive / xTB-unreliable groups that are NEUTRAL (charged ones — nitro, etc. —
# are already caught by the per-atom formal-charge rule). SMARTS → label.
REACTIVE_SMARTS = {
    "nitroso_N=O":      "[NX2]=[OX1]",
    "azo/diazo_N=N":    "[NX2]=[NX2]",
    "SF5_pentafluoro":  "[#16](F)(F)(F)(F)F",   # -SF5 pentafluorosulfanyl (explicit)
    "hypervalent_S-F":  "[#16](F)(F)(F)F",       # SF4 and SF5 (≥4 S–F); xTB-unreliable
    "ketene/cumulene":  "[CX2]=[CX2]=[#6,#8]",
    "isocyanide":       "[CX1-]#[NX2+]",
}
_REACTIVE = [(k, Chem.MolFromSmarts(v)) for k, v in REACTIVE_SMARTS.items()]


def classify(smiles: str) -> str:
    """Return 'keep' or a rejection reason."""
    if not isinstance(smiles, str) or not smiles.strip():
        return "invalid_parse"
    if "." in smiles:
        return "multi_component"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid_parse"
    if Chem.GetFormalCharge(mol) != 0:
        return "charged"
    if any(a.GetAtomicNum() not in ALLOWED_Z for a in mol.GetAtoms()):
        return "disallowed_element"
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return "isotope"
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):   # net 0 but charged atoms
        return "zwitterion_or_nitro"
    for name, patt in _REACTIVE:
        if patt is not None and mol.HasSubstructMatch(patt):
            return "reactive_group"
    if len(mol.GetSubstructMatches(TRUE_ALDEHYDE)) != 1:
        return "not_single_aldehyde"
    return "keep"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/library/aldehydes_clean.csv")
    ap.add_argument("--output", default="data/library/aldehydes_clean_v2.csv")
    ap.add_argument("--rejects", default="data/library/aldehydes_rejected_v2.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    smi_col = "SMILES" if "SMILES" in df.columns else "smiles"
    print(f"Loaded {len(df):,} rows from {args.input}")

    reasons = df[smi_col].map(classify)
    counts = Counter(reasons)
    print("\n=== Filter results ===")
    order = ["keep", "multi_component", "charged", "disallowed_element", "isotope",
             "zwitterion_or_nitro", "reactive_group", "not_single_aldehyde",
             "invalid_parse"]
    for k in order + [r for r in counts if r not in order]:
        n = counts.get(k, 0)
        print(f"  {k:20s} {n:7,d}  ({100*n/len(df):5.2f}%)")

    keep = df[reasons == "keep"].copy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(args.output, index=False)
    print(f"\nKept {len(keep):,} / {len(df):,} -> {args.output}")

    rej = df[reasons != "keep"].copy()
    rej["reject_reason"] = reasons[reasons != "keep"]
    rej.to_csv(args.rejects, index=False)
    print(f"Rejected {len(rej):,} -> {args.rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
