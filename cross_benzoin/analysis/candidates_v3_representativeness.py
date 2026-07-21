#!/usr/bin/env python3
"""
Is candidates_v3's 2M-pair sample representative of the FULL combinatorial
space (C(220859,2) ~= 24.4B pairs) it was drawn from? Never checked before.

Method: draw a FRESH independent random sample of pairs directly from the
full 220,859-aldehyde library (uniform over molecules, not restricted to
anything already in candidates_v3), compute Morgan/ECFP fingerprints for
both samples' donor+acceptor molecules, and compare:
  1. cho_class category-pair distribution (coarse check -- candidates_v3 is
     DELIBERATELY stratified evenly across 6 category-pairs, so this is
     expected to differ from the fresh sample's natural/unstratified mix;
     reported for context, not as a red flag by itself)
  2. per-molecule structural diversity (Bemis-Murcko scaffold counts, mean
     pairwise Tanimoto distance within each sample) -- THIS is the real
     representativeness question: does candidates_v3 cover a comparably
     diverse slice of scaffold-space as an unrestricted random draw, or is
     it narrower?

No DFT/xTB, no SLURM needed -- pure RDKit fingerprint ops on SMILES already
on disk. Fresh sample size matches candidates_v3's for a fair comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent.parent
ALD_LIB = REPO / "data/library/aldehydes_clean_v6.csv"
CANDIDATES_ALD = REPO / "data/cross_benzoin/candidates_v3/cross_benzoin_aldehydes_v3.csv.gz"
CANDIDATES_PAIRS = REPO / "data/cross_benzoin/candidates_v3/cross_benzoin_dG_candidates_v3.csv.gz"


def scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf is not None else None
    except Exception:
        return None


def fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def mean_pairwise_tanimoto_distance(fps: list, n_pairs: int, rng: np.random.Generator) -> float:
    """Estimate via random pair sampling (exact O(n^2) infeasible at this n)."""
    n = len(fps)
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    dists = []
    for a, b in zip(idx_a, idx_b):
        if a == b or fps[a] is None or fps[b] is None:
            continue
        sim = DataStructs.TanimotoSimilarity(fps[a], fps[b])
        dists.append(1.0 - sim)
    return float(np.mean(dists)) if dists else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fresh-molecules", type=int, default=5000,
                     help="molecules to sample for the fresh-draw comparison (fingerprint cost scales here)")
    ap.add_argument("--n-tanimoto-pairs", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ald_full = pd.read_csv(ALD_LIB, usecols=["InChIKey", "SMILES", "cho_class"])
    print(f"full library: {len(ald_full)} aldehydes")

    ald_v3 = pd.read_csv(CANDIDATES_ALD, low_memory=False)
    print(f"candidates_v3 aldehyde table: {len(ald_v3)} rows, cols={ald_v3.columns.tolist()[:10]}")
    smiles_col = "SMILES" if "SMILES" in ald_v3.columns else "smiles"
    key_col = "InChIKey" if "InChIKey" in ald_v3.columns else "InChIKey".lower()

    # candidates_v3 molecules ARE the full 220,859 (100% coverage per its own manifest) --
    # so the molecule-level set is identical by construction. The real question is about
    # the PAIR-level sample: which pairs actually appear in the 2M candidates_v3 pool vs a
    # fresh unrestricted random draw of pairs from the same full molecule set.
    pair_cols = ["donor_InChIKey", "acceptor_InChIKey", "donor_class", "acceptor_class"]
    v3_pairs = pd.read_csv(CANDIDATES_PAIRS, usecols=lambda c: c in pair_cols, low_memory=False)
    v3_pairs = v3_pairs.drop_duplicates(subset=["donor_InChIKey", "acceptor_InChIKey"])
    print(f"candidates_v3 unique directed pairs (dedup check): {len(v3_pairs)}")

    n_sample = min(args.n_fresh_molecules, len(v3_pairs))
    fresh_idx = rng.choice(len(ald_full), size=(n_sample, 2), replace=True)
    ald_lookup = ald_full.set_index("InChIKey")
    ald_smiles_arr = ald_full["SMILES"].to_numpy()
    ald_class_arr = ald_full["cho_class"].to_numpy()
    fresh_donor_smiles = ald_smiles_arr[fresh_idx[:, 0]]
    fresh_acceptor_smiles = ald_smiles_arr[fresh_idx[:, 1]]
    fresh_donor_class = ald_class_arr[fresh_idx[:, 0]]
    fresh_acceptor_class = ald_class_arr[fresh_idx[:, 1]]
    fresh_class_pair = ["__".join(sorted([str(a), str(b)]))
                         for a, b in zip(fresh_donor_class, fresh_acceptor_class)]

    v3_sample = v3_pairs.sample(n=min(n_sample, len(v3_pairs)), random_state=args.seed)
    v3_class_pair = v3_sample.apply(
        lambda r: "__".join(sorted([str(r["donor_class"]), str(r["acceptor_class"])])), axis=1)

    print("\n=== 1. class-pair distribution: candidates_v3 (stratified) vs fresh unrestricted draw ===")
    dist = pd.DataFrame({
        "candidates_v3_frac": v3_class_pair.value_counts(normalize=True),
        "fresh_unrestricted_frac": pd.Series(fresh_class_pair).value_counts(normalize=True),
    }).fillna(0)
    print(dist.to_string())
    dist.to_csv(args.outdir / "class_pair_distribution_comparison.csv")

    # ---------- 2. structural diversity: scaffold + fingerprint spread ----------
    print(f"\ncomputing fingerprints for {n_sample} fresh-draw molecules + "
          f"{n_sample} candidates_v3-sample molecules (donor side only, for tractability)...")
    fresh_mol_smiles = list(pd.unique(np.concatenate([fresh_donor_smiles, fresh_acceptor_smiles])))[:n_sample]
    v3_mol_ids = pd.unique(np.concatenate([v3_sample["donor_InChIKey"], v3_sample["acceptor_InChIKey"]]))[:n_sample]
    v3_mol_smiles = [ald_lookup["SMILES"].get(k) for k in v3_mol_ids]
    v3_mol_smiles = [s for s in v3_mol_smiles if isinstance(s, str)]

    fresh_fps = [fp(s) for s in fresh_mol_smiles]
    v3_fps = [fp(s) for s in v3_mol_smiles]
    fresh_scaffolds = {scaffold(s) for s in fresh_mol_smiles} - {None}
    v3_scaffolds = {scaffold(s) for s in v3_mol_smiles} - {None}

    fresh_div = mean_pairwise_tanimoto_distance(fresh_fps, args.n_tanimoto_pairs, rng)
    v3_div = mean_pairwise_tanimoto_distance(v3_fps, args.n_tanimoto_pairs, rng)

    result = {
        "n_fresh_molecules": len(fresh_mol_smiles),
        "n_v3_sample_molecules": len(v3_mol_smiles),
        "fresh_unique_scaffolds": len(fresh_scaffolds),
        "v3_unique_scaffolds": len(v3_scaffolds),
        "fresh_scaffold_frac": len(fresh_scaffolds) / max(1, len(fresh_mol_smiles)),
        "v3_scaffold_frac": len(v3_scaffolds) / max(1, len(v3_mol_smiles)),
        "fresh_mean_pairwise_tanimoto_distance": fresh_div,
        "v3_mean_pairwise_tanimoto_distance": v3_div,
    }
    print("\n=== 2. structural diversity comparison ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    pd.Series(result).to_csv(args.outdir / "diversity_comparison.csv")

    print(f"\nwrote representativeness check to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
