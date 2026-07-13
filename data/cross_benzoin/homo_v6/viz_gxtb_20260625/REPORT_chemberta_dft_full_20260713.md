# Fine-tuned ChemBERTa on full-library REAL DFT labels (2026-07-13)

Pretrained checkpoint: `seyonec/ChemBERTa-zinc-base-v1` (RoBERTa MLM, ~10M ZINC SMILES). Fine-tuned
end-to-end (not frozen-backbone) with a small regression head on the [CLS] token,
on the SAME task/data/split as `gnn_smiles_dft_full.py`: reactant (aldehyde)
SMILES only, target `dG_orca_kcal` (real DFT r2SCAN-3c), ALL cho_class
categories, seed=42 random 70:20:10 split.

- N total = 219,292, train = 153,505, val = 43,858, test = 21,929
- **Test MAE = 2.816 kcal/mol, RMSE = 4.203, R2 = 0.572**
- Reference: tabular champion MORDREDSLIM271_BDEGXTB MAE = 1.503; GNN+tabular
  blend MAE = 1.427; prior tiny-n (1,633) pure-SMILES attempts: SELFIES 3.16,
  ECFP+xgb 3.01, seq-GRU 3.25, RDKit-2D 2.92.

Params: 44,399,617 (12 max epochs,
early-stopped on val R2, patience 3 -- pretrained transformers converge much
faster than from-scratch GNNs). Full diagnostics: parity PNG + per-molecule test
predictions CSV in this dir.
