# Product BDFE pure-SMILES (no-xtb) surrogate (20260704)

Predicts product C-C BDFE (GFN2, 206,842 molecules) from ONLY 16 RDKit 2D descriptors computable directly from the SMILES string -- no conformer, no xtb at all. Compare to the QM-feature surrogate (`train_bde_surrogate.py`, needs xtb geometry+SP): R2=0.901, MAE=2.70.

- **pure-SMILES(2D)**: n_feat=16 n=206,842 MAE=7.155 RMSE=10.402 R2=0.597
- Top predictive feats: NumStereocenters, ArRings, FractionCSP3, HBA, Rings

## Interpretation

R2 gap vs the QM-feature surrogate: 0.304 -- **substantial (QM electronic-structure info is doing real work; not a cheap substitute)**.
