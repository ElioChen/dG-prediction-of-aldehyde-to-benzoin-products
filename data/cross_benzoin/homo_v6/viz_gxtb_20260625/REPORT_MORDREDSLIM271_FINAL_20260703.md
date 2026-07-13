# mordredslim271 production model (20260703)

- Features: **271** (72 champion QM + 199 targeted mordred: MoRSE/CPSA/Polarizability/GeometricalIndex/MomentOfInertia/PBF/McGowanVolume/VdwVolumeABC/Weight/TopoPSA)
- **TEST MAE 1.525, RMSE 2.305, R2 0.870** vs baseline_72 noise band 1.571+/-0.013 (REPORT_robustness_baseline72_20260702.md) -- a real improvement
- Uncertainty routing: confident 85% MAE 1.262 | routed MAE 3.016 | ROC AUC 0.799
- Model: `/scratch-shared/schen3/benzoin-dg/pipeline/models/gxtb_dft_correction_MORDREDSLIM271_20260703.joblib`
- Full-library output: `products_dG_corrected_MORDREDSLIM271_20260703.csv` (218,227 mols; 33,271 flagged route_to_dft)
