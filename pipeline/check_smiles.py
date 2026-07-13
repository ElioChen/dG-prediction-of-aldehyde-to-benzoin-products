#!/usr/bin/env python
"""
SMILES sanity check for the benzoin condensation workflow.

Validates every (reactant aldehyde, product benzoin) pair so malformed
structures never reach (or never silently survive) the DFT / conformer stage.

Benzoin condensation is atom-conserving:   2 R-CHO  ->  R-CO-CH(OH)-R
So the product formula MUST equal exactly 2x the aldehyde formula, the product
MUST contain the alpha-hydroxyketone motif, and the coupling MUST happen at a
*carbon-bound* aldehyde (not an N-/O-formyl, the known generator bug that leaves
a free R-CHO in the product -- see memory benzoin-generator-formyl-bug).

Failure policy (per user, 2026-06-20): a flagged pair is NOT trusted. The
product SMILES is regenerated with a carbon-CHO-constrained reaction and the
molecule is re-emitted for a fresh conformer search + relabel. Re-running the
conformer search alone is not enough when the SMILES itself is wrong.

Usage:
    # check a results dir of chunk_*.csv (or a single csv)
    python check_smiles.py RESULTS_DIR_OR_CSV [--ald-col aldehyde_smiles]
        [--bz-col benzoin_smiles] [--out failures.csv]
        [--reconform-input reconform.csv]

Columns expected (defaults match dft_sp_r2scan3c_full output):
    aldehyde_smiles, benzoin_smiles, and an id column (index / idx / PubChem_CID)
Exit code is non-zero if any pair fails, so it can gate a pipeline step.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")

# Carbon-bound aldehyde only -- this is the *intended* benzoin coupling site and
# the corrected reaction that excludes N-/O-formyl.
_REAL_CHO = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
# alpha-hydroxyketone core of a benzoin:  C(=O)-C(-OH)
_BENZOIN_MOTIF = Chem.MolFromSmarts("[#6][CX3](=[OX1])[CX4]([OX2H1])[#6,#1]")
# Corrected, site-constrained generator (carbon-bound CHO on both partners).
_BENZOIN_RXN_FIXED = AllChem.ReactionFromSmarts(
    "[CX3H1:1](=[O:2])[#6:5].[CX3H1:3](=[O:4])[#6:6]"
    ">>[C:5][C:1]([OH1:2])[C:3](=[O:4])[C:6]"
)


def _formula(mol: Chem.Mol) -> Counter:
    """Element counts incl. explicit H, order-independent."""
    m = Chem.AddHs(mol)
    return Counter(a.GetSymbol() for a in m.GetAtoms())


def _n_real_cho(mol: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(_REAL_CHO))


def check_pair(ald_smi: str, bz_smi: str) -> str:
    """Return '' if the pair is valid, else a short failure reason."""
    ald = Chem.MolFromSmiles(ald_smi or "")
    if ald is None:
        return "ald_parse_fail"
    if not bz_smi or not bz_smi.strip():
        return "bz_missing"
    bz = Chem.MolFromSmiles(bz_smi)
    if bz is None:
        return "bz_parse_fail"

    fa, fb = _formula(ald), _formula(bz)
    if fb != fa + fa:  # product must be exactly 2x aldehyde
        return "formula_mismatch"
    if not bz.HasSubstructMatch(_BENZOIN_MOTIF):
        return "no_benzoin_motif"

    n_ald = _n_real_cho(ald)
    if n_ald >= 1:
        # a correct benzoin consumes exactly two real C-CHO groups
        expected_free = 2 * n_ald - 2
        if _n_real_cho(bz) > expected_free:
            return "formyl_leftover"  # coupled at N/O-formyl, real CHO untouched
    return ""


def regen_benzoin(ald_smi: str) -> str | None:
    """Corrected product SMILES via carbon-CHO-constrained reaction."""
    mol = Chem.MolFromSmiles(ald_smi or "")
    if mol is None:
        return None
    try:
        for (p,) in _BENZOIN_RXN_FIXED.RunReactants((mol, mol)):
            try:
                Chem.SanitizeMol(p)
                return Chem.MolToSmiles(p)
            except Exception:
                continue
    except Exception:
        return None
    return None


def _iter_rows(path: str, ald_col: str, bz_col: str):
    files = (
        sorted(glob.glob(os.path.join(path, "chunk_*.csv")))
        if os.path.isdir(path)
        else [path]
    )
    if not files:
        sys.exit(f"no CSVs found at {path}")
    id_cols = ("index", "idx", "PubChem_CID", "aldehyde_name")
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                if ald_col not in row or bz_col not in row:
                    sys.exit(f"missing columns {ald_col!r}/{bz_col!r} in {f}")
                ident = {c: row.get(c, "") for c in id_cols if c in row}
                yield f, ident, row[ald_col], row.get(bz_col, "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="results dir of chunk_*.csv, or a single CSV")
    ap.add_argument("--ald-col", default="aldehyde_smiles")
    ap.add_argument("--bz-col", default="benzoin_smiles")
    ap.add_argument("--out", default="smiles_check_failures.csv")
    ap.add_argument(
        "--reconform-input",
        default=None,
        help="also write a re-run input CSV (index,aldehyde_smiles,"
        "benzoin_smiles_fixed) of fixable failures for a fresh conformer search",
    )
    args = ap.parse_args()

    total = 0
    reasons: Counter = Counter()
    failures = []  # (chunk, ident, ald, bz, reason)
    for chunk, ident, ald, bz in _iter_rows(args.path, args.ald_col, args.bz_col):
        total += 1
        reason = check_pair(ald, bz)
        if reason:
            reasons[reason] += 1
            failures.append((chunk, ident, ald, bz, reason))

    ok = total - sum(reasons.values())
    print(f"checked pairs   : {total}")
    print(f"valid           : {ok} ({100*ok/total:.2f}%)" if total else "no rows")
    print(f"failed          : {sum(reasons.values())} "
          f"({100*sum(reasons.values())/total:.2f}%)" if total else "")
    for r, c in reasons.most_common():
        print(f"  {c:7d}  {r}")

    if failures:
        cols = ["index", "PubChem_CID", "aldehyde_name", "aldehyde_smiles",
                "benzoin_smiles", "reason", "source_chunk"]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for chunk, ident, ald, bz, reason in failures:
                row = {c: ident.get(c, "") for c in cols}
                row.update(aldehyde_smiles=ald, benzoin_smiles=bz,
                           reason=reason, source_chunk=os.path.basename(chunk))
                w.writerow(row)
        print(f"\nwrote failure list -> {args.out}")

        if args.reconform_input:
            n = 0
            with open(args.reconform_input, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["index", "aldehyde_smiles", "benzoin_smiles_fixed",
                            "orig_reason"])
                for _chunk, ident, ald, _bz, reason in failures:
                    if reason in ("ald_parse_fail",):
                        continue  # upstream input problem, not regenerable
                    fixed = regen_benzoin(ald)
                    if fixed:
                        w.writerow([ident.get("index", ""), ald, fixed, reason])
                        n += 1
            print(f"wrote {n} re-conformer inputs -> {args.reconform_input}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
