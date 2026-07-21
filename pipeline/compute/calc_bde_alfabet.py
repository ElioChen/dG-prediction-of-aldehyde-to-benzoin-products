#!/usr/bin/env python
"""ALFABET baseline BDE/BDFE for the same two bonds the project's own g-xTB pipeline
targets: the aldehyde formyl C-H bond and the product's new ketC-carbC bond (see
calc_bde_free_energy_gxtb.py). Purely 2D (SMILES-only) -- no xyz/xtb required -- so this
is meant as a fast, generic baseline to compare against the project's own BDE/BDFE
descriptors (BDE_prediction.md, Phase 1: "run ALFABET on existing data").

ALFABET (NREL, pstjohn/alfabet) pins scikit-learn==0.24.2 and needs an old TF/nfp stack
that conflicts with this project's modern rdkit/numpy -- run this ONLY with the isolated
venv at /gpfs/scratch1/shared/schen3/envs/alfabet (see pipeline/gnn/README.md for the
established pattern of dedicated envs for heavy/conflicting deps). Do not pip install
alfabet into the main project env or bde_lite/gnn envs.

Bond identification is done independently in this script via RDKit SMARTS, then matched
to ALFABET's own (molecule=canonical_smiles, bond_index) key -- see
alfabet.fragment.Molecule/fragment_iterator: bond_index is bond.GetIdx() on
Chem.AddHs(Chem.MolFromSmiles(smiles)). Molecule.smiles always re-canonicalizes
(MolToSmiles(self.mol)), and predict() passes that canonical string right back into a
fresh Molecule(smiles=...), so alfabet's own bond_index is always computed from
Chem.MolFromSmiles(CANONICAL_smiles) -- NOT from whatever atom order the input SMILES
happened to use. Source SMILES in this project's CSVs are frequently NOT already RDKit-
canonical (e.g. Kekulized "C1=CC=C...C=O" rather than canonical "c1ccccc1..."), so this
script must canonicalize FIRST and compute c_idx/h_idx/bond_index from a mol parsed from
that canonical string -- computing them from the original (possibly differently-ordered)
parse and pairing them with the canonical string, as an earlier version of this script
did, silently attaches the correct canonical SMILES to the WRONG bond_index whenever atom
order shifts under canonicalization (verified: this happened for a large, unknown
fraction of the aldehyde library -- e.g. id 52677.0's formyl C-H bond was bond_index 34
under its original atom order but bond_index 30 under canonical order, so the original
code fetched ALFABET's prediction for an unrelated ring C-H instead).

PILOT usage:
  ENV=/gpfs/scratch1/shared/schen3/envs/alfabet
  $ENV/bin/python calc_bde_alfabet.py --which aldehydes --n 50 --out /tmp/alfabet_pilot_ald.csv
  $ENV/bin/python calc_bde_alfabet.py --which products --n 50 --out /tmp/alfabet_pilot_prod.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

FORMYL_CH = Chem.MolFromSmarts("[CX3H1](=O)")
BENZOIN_CORE = Chem.MolFromSmarts("[#6](=O)[#6]([OX2H1])")

H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")


def target_bond_aldehyde(smiles):
    """Return (canonical_smiles, bond_index) for the formyl C-H bond, or None.

    Must canonicalize BEFORE computing indices: alfabet's own bond_index is always
    derived from Chem.MolFromSmiles(canonical_smiles), which can order atoms/bonds
    differently than the source SMILES (see module docstring)."""
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return None
    smiles = Chem.MolToSmiles(mol0)
    mol = Chem.MolFromSmiles(smiles)
    match = mol.GetSubstructMatch(FORMYL_CH)
    if not match:
        return None
    c_idx = match[0]
    molH = Chem.AddHs(mol)
    h_idx = next((nbr.GetIdx() for nbr in molH.GetAtomWithIdx(c_idx).GetNeighbors()
                  if nbr.GetSymbol() == "H"), None)
    if h_idx is None:
        return None
    bond = molH.GetBondBetweenAtoms(c_idx, h_idx)
    if bond is None:
        return None
    return smiles, bond.GetIdx()


def target_bond_product(smiles):
    """Return (canonical_smiles, bond_index) for the ketC-carbC bond, or None.

    Must canonicalize BEFORE computing indices -- see target_bond_aldehyde."""
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return None
    smiles = Chem.MolToSmiles(mol0)
    mol = Chem.MolFromSmiles(smiles)
    match = mol.GetSubstructMatch(BENZOIN_CORE)
    if not match:
        return None
    ket_c, _ket_o, carb_c, _hyd_o = match
    molH = Chem.AddHs(mol)
    bond = molH.GetBondBetweenAtoms(ket_c, carb_c)
    if bond is None:
        return None
    return smiles, bond.GetIdx()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["aldehydes", "products"], required=True)
    ap.add_argument("--n", type=int, default=50, help="pilot sample size (None = full file)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    finder = target_bond_aldehyde if args.which == "aldehydes" else target_bond_product
    src = H / f"{args.which}_all.csv"
    df = pd.read_csv(src, usecols=["id", "smiles", "error"], dtype=str, keep_default_na=False)
    df = df[(df["error"] == "") & (df["smiles"] != "")].drop_duplicates("id").reset_index(drop=True)
    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    rows, n_unmatched = [], 0
    for _id, smi in zip(df["id"], df["smiles"]):
        hit = finder(smi)
        if hit is None:
            n_unmatched += 1
            continue
        csmi, bidx = hit
        rows.append((_id, csmi, bidx))
    matched = pd.DataFrame(rows, columns=["id", "molecule", "bond_index"])
    print(f"{args.which}: {len(matched)}/{len(df)} matched target bond "
          f"({n_unmatched} unmatched)", flush=True)

    # De-duplicate BEFORE calling predict: duplicate canonical SMILES in the input list
    # crashes alfabet.model.predict's own internal pd.DataFrame(...).T.unstack().reindex()
    # with "cannot handle a non-unique multi-index" (hit in production on the products
    # side after a full 220k-row, 6.5h run -- only surfaces with certain duplicate
    # patterns, didn't happen on the aldehyde side). De-duplicating here and merging back
    # by (molecule, bond_index) is also strictly MORE correct than the previous version,
    # which silently dropped every id but the last whenever two ids shared a canonical
    # SMILES; now every id gets its (identical, correctly shared) prediction.
    canon_unique = sorted(matched["molecule"].unique())
    n_dupes = len(matched) - len(canon_unique)
    if n_dupes:
        print(f"{n_dupes} rows share a canonical SMILES with another row "
              f"({len(canon_unique)} unique molecules to predict)", flush=True)

    import alfabet.model as alfabet_model
    pred = alfabet_model.predict(canon_unique, drop_duplicates=False, batch_size=32,
                                  verbose=True)

    out = matched.merge(pred, on=["molecule", "bond_index"], how="left")
    out = out[out["bde_pred"].notna()][
        ["id", "molecule", "bond_type", "fragment1", "fragment2",
         "bde_pred", "bdfe_pred", "is_valid"]
    ].rename(columns={
        "molecule": "smiles_canonical",
        "bde_pred": "bde_alfabet_kcal",
        "bdfe_pred": "bdfe_alfabet_kcal",
    })
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  {len(out)}/{len(matched)} rows recovered", flush=True)


if __name__ == "__main__":
    main()
