# Pure-SMILES GINE on full-library REAL DFT labels (2026-07-13)

Reactant (aldehyde) SMILES only, no QM/xTB descriptors. Target: `dG_orca_kcal`
(real DFT r2SCAN-3c label, `data/raw/dft_sp_funnelv3/dft_labels_all.parquet`),
ALL cho_class categories, seed=42 random 70:20:10 split (same protocol as the
tabular/GNN champion).

- N total = 219,292, train = 153,505, val = 43,858, test = 21,929
- **Test MAE = 2.671 kcal/mol, RMSE = 4.018, R2 = 0.608**
- Reference: tabular champion MORDREDSLIM271_BDEGXTB (72 QM + 199 mordred + 4
  BDE/BDFE) test MAE = 1.503; GNN+tabular blend test MAE = 1.427.
- Prior pure-SMILES attempts (n=1,633 only): SELFIES counts 3.16, ECFP+xgb 3.01,
  seq-GRU 3.25, vs RDKit-2D descriptor surrogate ~2.92 (all far short of the
  Δ-learning tabular floor).

Params: 238,337. Full diagnostics: parity/
residual/training-curve PNGs + per-molecule test predictions CSV in this dir.
