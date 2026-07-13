# Mordred-augmented feature comparison (20260703)

Same 70:20:10 split (seed 42), same ensemble (MLP+XGB8+XGB10), vs baseline_72 noise band 1.571+/-0.013 (see REPORT_robustness_baseline72_20260702.md). 4/1099 product chunks (~800 mols, ~0.36%) timed out and are simply missing (dropna), not backfilled, per instruction to proceed with what's done.

- **baseline_72**: n_feat=72 n=219,095 MAE=1.577 RMSE=2.399 R2=0.859 scope={'aromatic': 1.3825294515663162, 'aliphatic': 1.9855824911644817}
- **72_plus_mordred438**: n_feat=510 n=219,095 MAE=1.535 RMSE=2.308 R2=0.869 scope={'aromatic': 1.3647957605685193, 'aliphatic': 1.8938361938003763}
