# FINAL g-xTB->DFT correction (20260626)

- DFT labels: **219,364/219,421** (100%); training set 219,095
- Model: **ensemble MLP+3×XGB on 72 feats** (56 QM + 16 RDKit global), Δ-learning
- **TEST MAE 1.566, RMSE 2.380, R² 0.861**
- Uncertainty routing: flag most-uncertain 15% (std≥10.57) -> route to DFT
  - confident 85% MAE **1.280** | routed MAE 3.185
- Scope: {'aromatic': {'MAE': 1.370761505474867, 'conf_MAE': 1.1871536372769302}, 'aliphatic': {'MAE': 1.9746687043362228, 'conf_MAE': 1.5232375847869726}}
- Model: `/scratch-shared/schen3/benzoin-dg/pipeline/models/gxtb_dft_correction_ENSEMBLE72_20260626.joblib`
- Full-library output: `products_dG_corrected_FINAL_20260626.csv` (218,227 mols; 32,774 flagged route_to_dft)
