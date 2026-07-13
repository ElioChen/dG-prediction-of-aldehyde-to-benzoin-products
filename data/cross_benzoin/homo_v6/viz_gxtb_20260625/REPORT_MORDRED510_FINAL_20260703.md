# mordred510 production model (20260703)

- Features: **510** (72 champion QM + 438 targeted mordred: MoRSE/CPSA/Polarizability/GeometricalIndex/MomentOfInertia/PBF/McGowanVolume/VdwVolumeABC/Weight/TopoPSA)
- **TEST MAE 1.517, RMSE 2.290, R2 0.871** vs baseline_72 noise band 1.571+/-0.013 (REPORT_robustness_baseline72_20260702.md) -- a real improvement
- Uncertainty routing: confident 85% MAE 1.259 | routed MAE 2.983 | ROC AUC 0.799
- Model: `/scratch-shared/schen3/benzoin-dg/pipeline/models/gxtb_dft_correction_MORDRED510_20260703.joblib`
- Full-library output: `products_dG_corrected_MORDRED510_20260703.csv` (218,227 mols; 32,768 flagged route_to_dft)
