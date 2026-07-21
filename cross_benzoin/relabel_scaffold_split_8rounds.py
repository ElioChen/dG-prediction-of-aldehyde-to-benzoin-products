#!/usr/bin/env python
"""
Re-derive pair-level scaffold-disjoint labels (new_scaffold_split) for the round1-8
combined table, against the PRODUCTION (80/10/10) split -- see memory
cross-split-ratio-decision-20260720.md for why 80/10/10, not 70/20/10, is production.
Same derivation rule as the original round1-7 labeling (verified 2026-07-20): a pair is
train/validation/test only if donor AND acceptor land in the same molecule-level
scaffold_split; otherwise "mixed" (excluded from both train and test).
"""
import pandas as pd
from rdkit import Chem

ald = pd.read_parquet("data/cross_benzoin/candidates_v3/aldehydes_with_scaffold_split.parquet")
pairs = pd.read_parquet("data/cross_benzoin/cross_round8/cross_train_table_8rounds_mordred.parquet")
print("pairs shape:", pairs.shape)

ald["canon"] = ald["SMILES"].apply(lambda s: Chem.CanonSmiles(s) if Chem.MolFromSmiles(s) else None)
split_lut = dict(zip(ald["canon"], ald["scaffold_split"]))

pairs["donor_canon"] = pairs["donor_smiles"].apply(lambda s: Chem.CanonSmiles(s) if isinstance(s, str) and Chem.MolFromSmiles(s) else None)
pairs["acceptor_canon"] = pairs["acceptor_smiles"].apply(lambda s: Chem.CanonSmiles(s) if isinstance(s, str) and Chem.MolFromSmiles(s) else None)

pairs["donor_scaf_split"] = pairs["donor_canon"].map(split_lut)
pairs["acceptor_scaf_split"] = pairs["acceptor_canon"].map(split_lut)

matched = pairs["donor_scaf_split"].notna() & pairs["acceptor_scaf_split"].notna()
print(f"matched {matched.sum()}/{len(pairs)} pairs to library scaffold_split")

same = pairs["donor_scaf_split"] == pairs["acceptor_scaf_split"]
pairs["new_scaffold_split"] = pairs["donor_scaf_split"].where(same & matched, "mixed")
pairs.loc[~matched, "new_scaffold_split"] = "mixed"

vc = pairs["new_scaffold_split"].value_counts()
print(vc)
print((vc / len(pairs) * 100).round(1))

out = "data/cross_benzoin/cross_round8/cross_train_table_8rounds_scaffold_split_labeled.parquet"
pairs.to_parquet(out)
print("wrote", out)
