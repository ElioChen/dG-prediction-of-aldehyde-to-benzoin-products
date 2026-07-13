# MORDREDSLIM271_BDEGXTB production model (20260706)

- Features: **275** (mordredslim271's 271 [72 champion QM + 199 targeted mordred] + 4 g-xTB-consistent BDE/BDFE: prod_bdfe_gxtb_kcal, ald_bdfe_gxtb_kcal, prod_bde_gxtb_kcal, ald_bde_gxtb_kcal)
- **TEST MAE 1.503, RMSE 2.257, R2 0.875** vs mordredslim271 production champion (test MAE 1.525) and the quick 2-member XGB check that motivated this run (1.612 -> 1.563, REPORT_bdfe_gxtb_full_augment_20260706.md)
- Uncertainty routing: confident 85% MAE 1.252 | routed MAE 2.923 | ROC AUC 0.796
- Model: `/scratch-shared/schen3/benzoin-dg/pipeline/models/gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib`
- Full-library output: `products_dG_corrected_MORDREDSLIM271_BDEGXTB_20260706.csv` (218,227 mols; 33,275 flagged route_to_dft)
