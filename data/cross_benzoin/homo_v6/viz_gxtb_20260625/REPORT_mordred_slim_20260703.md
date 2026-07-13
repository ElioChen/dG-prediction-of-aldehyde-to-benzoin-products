# mordred510 SHAP importance + correlation slim (20260703)

Bundle: `gxtb_dft_correction_MORDRED510_20260703.joblib` (test MAE 1.517). SHAP on 4000-row test subsample (XGB_d8).

## Slimming result

kept **199/438** mordred feats (dropped 239 redundant, |corr|>0.9 with a higher-ranked kept feat). 72 champion QM feats never touched.

**Slim total: 271 feats** (72 + 199).

## Top-25 global importance

| rank | feature | mean|SHAP| | family |
|---|---|---|---|
| 1 | P_int | 1.9389 | QM(72) |
| 2 | ald_wbo_CO | 0.8308 | QM(72) |
| 3 | ald_P_int | 0.6253 | QM(72) |
| 4 | ald_xtb_dipole | 0.6061 | QM(72) |
| 5 | ald_pa_CHO_O | 0.4819 | QM(72) |
| 6 | xtb_dipole | 0.4084 | QM(72) |
| 7 | wbo_CC_new | 0.3377 | QM(72) |
| 8 | vbur_ketC | 0.3215 | QM(72) |
| 9 | mulliken_carbC | 0.2774 | QM(72) |
| 10 | ald_mordred_TPSA | 0.2439 | mordred |
| 11 | fukui_plus_ketC | 0.2180 | QM(72) |
| 12 | ald_mulliken_CHO_C | 0.2157 | QM(72) |
| 13 | mordred_TPSA | 0.2052 | mordred |
| 14 | ald_mordred_Mor02se | 0.1600 | mordred |
| 15 | ald_mulliken_CHO_O | 0.1495 | QM(72) |
| 16 | ald_mordred_WNSA3 | 0.1460 | mordred |
| 17 | mordred_GeomRadius | 0.1458 | mordred |
| 18 | ald_mordred_PNSA3 | 0.1418 | mordred |
| 19 | mordred_Mor03v | 0.1356 | mordred |
| 20 | ald_SASA_total | 0.1349 | QM(72) |
| 21 | mordred_TopoPSA(NO) | 0.1347 | mordred |
| 22 | mordred_Mor02m | 0.1273 | mordred |
| 23 | mulliken_ketC | 0.1189 | QM(72) |
| 24 | wbo_CO_carb | 0.1161 | QM(72) |
| 25 | mordred_Mor04m | 0.1115 | mordred |

Selection saved: `mordred_slim_selection_20260703.json` (use `slim_feats_total` to retrain a slimmed model and confirm it holds the MAE 1.517 level with fewer feats).
