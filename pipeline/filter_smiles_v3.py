#!/usr/bin/env python3
"""
Deep SMILES filter v3 for benzoin aldehyde candidates  (supersedes filter_smiles.py).

Same goal as v2 — keep only valid, single-aldehyde benzoin substrates clean enough
for the xTB/DFT ΔG pipeline — but RELAXED where the v2 cuts were over-strict and
discarding genuine substrates rather than genuinely problematic chemistry.

What changed vs v2 (and why)
  • ELEMENTS: allow B, Si, P, Se in addition to H,C,N,O,F,S,Cl,Br,I. All four are
    fully GFN2-xTB-parametrized and inert to the benzoin step (silyl/boronate/
    phosphonate/selenophene building blocks). Recovers ~7.3k molecules.
  • CHARGES: the v2 "any atom with a formal charge -> reject" rule was throwing away
    NITRO aldehydes (4-nitrobenzaldehyde is a textbook NHC/benzoin substrate; the
    [N+](=O)[O-] is just a formal-charge drawing of a NEUTRAL group). v3 whitelists
    the charge-separated motifs that are benzoin-compatible — nitro, organic azide,
    aromatic N-oxide, amine-oxide/nitrone — and rejects only the genuine
    ylides/mesoionics (an unexplained carbanion/heteroatom-anion) as
    'zwitterion_or_ylide'.

What stayed rejected (unchanged from v2)
  reactive_group : nitroso (N=O), azo/diazo (N=N), hypervalent S-F (SF4/SF5),
                   ketene/cumulene, isocyanide  — reactive and/or unphysical xTB
                   geometries; would not survive the reaction.
  isotope, multi_component, net-charged, not_single_aldehyde, invalid_parse.

xTB-noise tag (kept, not discarded)
  Nitro / azide / N-oxide are benzoin-OK but carried 2-3x the xTB->DFT correction
  noise in the n=1500 diagnostic (analyze_energies.py). Rather than exclude them we
  KEEP and TAG: column `xtb_risk` flags these so they can be down-weighted or given
  extra DFT scrutiny downstream.

Output columns added to the clean set
  cho_class  : aromatic_carbo / aromatic_hetero / vinyl_conj / aliphatic
               (single source of truth = cho_category.cho_class; the downstream
               selectors filter on this, so we do NOT fork the aromatic/aliphatic
               split here — one tagged file, categorized.)
  xtb_risk   : '' or a comma-list of high-noise motifs present (nitro/azide/n_oxide).

Usage
  # default: reconstruct the full raw pool (clean v1 + its element/charge rejects)
  python pipeline/filter_smiles_v3.py
  python pipeline/filter_smiles_v3.py --input a.csv b.csv --output clean.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cho_category import cho_class  # noqa: E402  single source of truth

RDLogger.DisableLog("rdApp.*")

# H C N O F  Si P S Cl  Se Br I   (+B)
ALLOWED_Z = {1, 6, 7, 8, 9, 5, 14, 15, 16, 17, 34, 35, 53}
TRUE_ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=O)[#6]")  # R-CHO, R = carbon

# Charge-separated motifs that are NEUTRAL overall AND benzoin-compatible. Every
# formally-charged atom in a kept molecule must be "explained" by one of these;
# otherwise it is a genuine ylide / mesoionic -> reject.
ALLOWED_CHARGED_SMARTS = {
    "nitro":        "[NX3+](=[OX1])[O-]",
    "azide_a":      "[NX2]=[NX2+]=[NX1-]",
    "azide_b":      "[NX1-]=[NX2+]=[NX2]",
    "n_oxide_arom": "[n+][O-]",
    "amine_oxide":  "[NX4+,NX3+][O-]",   # amine oxide + nitrone N-oxide
}
_ALLOWED_CHARGED = [Chem.MolFromSmarts(v) for v in ALLOWED_CHARGED_SMARTS.values()]

# Reactive / xTB-unreliable NEUTRAL groups (unchanged from v2).
REACTIVE_SMARTS = {
    "nitroso_N=O":      "[NX2]=[OX1]",
    "azo/diazo_N=N":    "[NX2]=[NX2]",
    "SF5_pentafluoro":  "[#16](F)(F)(F)(F)F",
    "hypervalent_S-F":  "[#16](F)(F)(F)F",
    "ketene/cumulene":  "[CX2]=[CX2]=[#6,#8]",
    "isocyanide":       "[CX1-]#[NX2+]",
}
_REACTIVE = [(k, Chem.MolFromSmarts(v)) for k, v in REACTIVE_SMARTS.items()]

# High-xTB-noise but kept motifs -> emitted in the `xtb_risk` tag column.
RISK_SMARTS = {
    "nitro":   "[NX3+](=[OX1])[O-]",
    "azide":   "[NX2,NX1-]=[NX2+]=[NX1-,NX2]",
    "n_oxide": "[$([n+][O-]),$([N+;!$([N+]=O)][O-])]",  # arom N-oxide/amine-oxide, not nitro
}
_RISK = [(k, Chem.MolFromSmarts(v)) for k, v in RISK_SMARTS.items()]


def _charged_ok(mol) -> bool:
    """True if every formally-charged atom belongs to an allowed charged motif."""
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()}
    if not charged:
        return True
    explained: set[int] = set()
    for patt in _ALLOWED_CHARGED:
        for match in mol.GetSubstructMatches(patt):
            explained.update(match)
    return charged.issubset(explained)


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
    if not _charged_ok(mol):
        return "zwitterion_or_ylide"
    for _name, patt in _REACTIVE:
        if patt is not None and mol.HasSubstructMatch(patt):
            return "reactive_group"
    if len(mol.GetSubstructMatches(TRUE_ALDEHYDE)) != 1:
        return "not_single_aldehyde"
    return "keep"


def xtb_risk(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return ",".join(k for k, p in _RISK if p is not None and mol.HasSubstructMatch(p))


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    lib = repo / "data/library"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+",
                    default=[str(lib / "aldehydes_clean.csv"),
                             str(lib / "aldehydes_rejected.csv")],
                    help="one or more CSVs; concatenated + de-duped on SMILES so the "
                         "element/charge molecules v1 rejected are recovered")
    ap.add_argument("--output", default=str(lib / "aldehydes_clean_v3.csv"))
    ap.add_argument("--rejects", default=str(lib / "aldehydes_rejected_v3.csv"))
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in args.input]
    df = pd.concat(frames, ignore_index=True)
    smi_col = "SMILES" if "SMILES" in df.columns else "smiles"
    if "reject_reason" in df.columns:
        df = df.drop(columns=["reject_reason"])
    before = len(df)
    df = df.drop_duplicates(subset=[smi_col]).reset_index(drop=True)
    print(f"Loaded {before:,} rows from {len(args.input)} file(s) -> "
          f"{len(df):,} unique SMILES")

    reasons = df[smi_col].map(classify)
    counts = Counter(reasons)
    print("\n=== Filter v3 results ===")
    order = ["keep", "multi_component", "charged", "disallowed_element", "isotope",
             "zwitterion_or_ylide", "reactive_group", "not_single_aldehyde",
             "invalid_parse"]
    for k in order + [r for r in counts if r not in order]:
        n = counts.get(k, 0)
        print(f"  {k:20s} {n:8,d}  ({100*n/len(df):5.2f}%)")

    keep = df[reasons == "keep"].copy()
    keep["cho_class"] = keep[smi_col].map(cho_class)
    keep["xtb_risk"] = keep[smi_col].map(xtb_risk)
    print("\n=== Kept by cho_class ===")
    for c, n in keep["cho_class"].value_counts().items():
        print(f"  {c:18s} {n:8,d}")
    n_risk = (keep["xtb_risk"] != "").sum()
    print(f"  [xtb_risk tagged: {n_risk:,} / {len(keep):,}]")

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
