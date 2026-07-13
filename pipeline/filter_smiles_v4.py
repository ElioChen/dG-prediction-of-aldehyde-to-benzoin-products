#!/usr/bin/env python3
"""
Benzoin aldehyde filter v4 — single-source filter from the RAW pool.

Earlier filters ran in two opaque stages (an upstream inclusion/exclusion pass that
produced data/library/aldehydes_clean.csv, then filter_smiles.py v2/v3). v4 collapses
everything into ONE transparent, reproducible pass directly on the raw dump
  /scratch-shared/schen3/aldehydes.csv   (~450k molecules, name/SMILES/MW/CID/...)
so the full provenance lives in one file. v2/v3 and their outputs are left untouched.

It combines:
  (A) the chemically-motivated inclusion/exclusion criteria for NHC benzoin substrates
  (B) the v3 relaxed element/charge logic (keep nitro/azide/N-oxide + B/Si/P/Se)

Rejection taxonomy (first failing rule wins)
  invalid_parse        RDKit cannot parse
  multi_component      contains '.'  (salt / mixture / counter-ion)
  net_charged          net formal charge != 0
  mw_too_high          MW > --max-mw (default 500 Da; xTB/DFT tractability)
  disallowed_element   element outside {H,C,N,O,F,S,Cl,Br,I, B,Si,P,Se}
                       (= "no metals" + drop exotic Te/Ge/As/Sb)
  isotope              any isotope-labelled atom (redundant for ΔG/descriptors)
  zwitterion_or_ylide  a per-atom formal charge NOT explained by an allowed motif
                       (nitro / azide / aromatic-N-oxide / amine-oxide are allowed)
  not_single_aldehyde  zero TRUE aldehydes [CX3H1](=O)[#6]  (formic/formyl/acyl/CH2O)
  multi_aldehyde       >1 TRUE aldehyde   (crossed/intramolecular benzoin)
  enal                 α,β-unsaturated aldehyde, non-aromatic C=C  (NHC→Stetter/homoenolate)
  ynal                 α,β-ynal                                     (same diversion)
  alpha_dicarbonyl     R-CO-CHO / glyoxylate ester / glyoxylamide   (adjacent C=O dominates)
  reactive_group       nitroso, azo/diazo, SF5/hypervalent S-F, ketene/cumulene, isocyanide
  aliphatic_too_large  no aromatic ring AND carbon count > --max-aliphatic-c (default 12)

Kept rows get two extra columns
  cho_class  aromatic_carbo / aromatic_hetero / vinyl_conj / aliphatic
             (single source of truth = cho_category.cho_class; the chemically decisive
             CHO-environment axis, used by the downstream selectors)
  xtb_risk   '' or comma-list of high xTB→DFT-noise motifs present (nitro/azide/n_oxide):
             KEPT but flagged so they can be down-weighted / given extra DFT scrutiny.

Usage
  python pipeline/filter_smiles_v4.py
  python pipeline/filter_smiles_v4.py --input /scratch-shared/schen3/aldehydes.csv \
         --max-mw 500 --max-aliphatic-c 12
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

ALLOWED_Z = {1, 6, 7, 8, 9, 5, 14, 15, 16, 17, 34, 35, 53}  # CHNOF SiP SCl SeBrI + B

TRUE_ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=O)[#6]")      # R-CHO, R = carbon
ENAL          = Chem.MolFromSmarts("[CX3H1](=O)[C;!a]=[C;!a]")   # non-aromatic α,β-C=C (incl. carbocyclic)
YNAL          = Chem.MolFromSmarts("[CX3H1](=O)[CX2]#[CX2]")
ALPHA_DICARB  = Chem.MolFromSmarts("[CX3H1](=O)[CX3](=O)[!#1]")  # keto/ester/amide-CHO

# Charge-separated motifs that are NEUTRAL overall AND benzoin-compatible; every
# formally-charged atom in a kept molecule must be explained by one of these.
ALLOWED_CHARGED_SMARTS = {
    "nitro":        "[NX3+](=[OX1])[O-]",
    "azide_a":      "[NX2]=[NX2+]=[NX1-]",
    "azide_b":      "[NX1-]=[NX2+]=[NX2]",
    "n_oxide_arom": "[n+][O-]",
    "amine_oxide":  "[NX4+,NX3+][O-]",
}
_ALLOWED_CHARGED = [Chem.MolFromSmarts(v) for v in ALLOWED_CHARGED_SMARTS.values()]

REACTIVE_SMARTS = {
    "nitroso_N=O":      "[NX2]=[OX1]",
    "azo/diazo_N=N":    "[NX2]=[NX2]",
    "SF5_pentafluoro":  "[#16](F)(F)(F)(F)F",
    "hypervalent_S-F":  "[#16](F)(F)(F)F",
    "ketene/cumulene":  "[CX2]=[CX2]=[#6,#8]",
    "isocyanide":       "[CX1-]#[NX2+]",
}
_REACTIVE = [(k, Chem.MolFromSmarts(v)) for k, v in REACTIVE_SMARTS.items()]

RISK_SMARTS = {
    "nitro":   "[NX3+](=[OX1])[O-]",
    "azide":   "[NX2,NX1-]=[NX2+]=[NX1-,NX2]",
    "n_oxide": "[$([n+][O-]),$([N+;!$([N+]=O)][O-])]",
}
_RISK = [(k, Chem.MolFromSmarts(v)) for k, v in RISK_SMARTS.items()]


def _charged_ok(mol) -> bool:
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()}
    if not charged:
        return True
    explained: set[int] = set()
    for patt in _ALLOWED_CHARGED:
        for match in mol.GetSubstructMatches(patt):
            explained.update(match)
    return charged.issubset(explained)


def classify(smiles, mw, max_mw: float, max_aliph_c: int) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        return "invalid_parse"
    if "." in smiles:
        return "multi_component"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid_parse"
    if Chem.GetFormalCharge(mol) != 0:
        return "net_charged"
    try:
        if float(mw) > max_mw:
            return "mw_too_high"
    except (TypeError, ValueError):
        pass
    if any(a.GetAtomicNum() not in ALLOWED_Z for a in mol.GetAtoms()):
        return "disallowed_element"
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return "isotope"
    if not _charged_ok(mol):
        return "zwitterion_or_ylide"
    n_ald = len(mol.GetSubstructMatches(TRUE_ALDEHYDE))
    if n_ald == 0:
        return "not_single_aldehyde"
    if n_ald > 1:
        return "multi_aldehyde"
    if mol.HasSubstructMatch(ENAL):
        return "enal"
    if mol.HasSubstructMatch(YNAL):
        return "ynal"
    if mol.HasSubstructMatch(ALPHA_DICARB):
        return "alpha_dicarbonyl"
    for _name, patt in _REACTIVE:
        if patt is not None and mol.HasSubstructMatch(patt):
            return "reactive_group"
    if not any(a.GetIsAromatic() for a in mol.GetAtoms()):
        n_c = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
        if n_c > max_aliph_c:
            return "aliphatic_too_large"
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
    ap.add_argument("--input", default="/scratch-shared/schen3/aldehydes.csv")
    ap.add_argument("--output", default=str(lib / "aldehydes_clean_v4.csv"))
    ap.add_argument("--rejects", default=str(lib / "aldehydes_rejected_v4.csv"))
    ap.add_argument("--max-mw", type=float, default=500.0)
    ap.add_argument("--max-aliphatic-c", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    smi_col = "SMILES" if "SMILES" in df.columns else "smiles"
    mw_col = "MW" if "MW" in df.columns else None
    before = len(df)
    df = df.drop_duplicates(subset=[smi_col]).reset_index(drop=True)
    print(f"Loaded {before:,} rows from {args.input} -> {len(df):,} unique SMILES")
    print(f"max_mw={args.max_mw}  max_aliphatic_c={args.max_aliphatic_c}")

    mw_series = df[mw_col] if mw_col else pd.Series([float('nan')] * len(df))
    reasons = pd.Series(
        [classify(s, m, args.max_mw, args.max_aliphatic_c)
         for s, m in zip(df[smi_col], mw_series)],
        index=df.index,
    )
    counts = Counter(reasons)
    print("\n=== Filter v4 results ===")
    order = ["keep", "multi_component", "net_charged", "mw_too_high",
             "disallowed_element", "isotope", "zwitterion_or_ylide",
             "not_single_aldehyde", "multi_aldehyde", "enal", "ynal",
             "alpha_dicarbonyl", "reactive_group", "aliphatic_too_large",
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
    print(f"  [xtb_risk tagged: {(keep['xtb_risk'] != '').sum():,} / {len(keep):,}]")

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
