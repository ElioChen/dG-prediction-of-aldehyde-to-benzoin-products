# MORDREDSLIM271_BDEGXTB SHAP importance + cost-aware slimming (20260707)

Bundle: `gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib` (test MAE 1.503). SHAP on 4000-row test subsample (XGB_d8).

## BDE vs BDFE: is the expensive descriptor pulling its weight?

| feature | SHAP rank (of 275) | mean|SHAP| | acquisition cost |
|---|---|---|---|
| prod_bdfe_gxtb_kcal | 15 | 0.1910 | EXPENSIVE (--ohess Hessian+RRHO) |
| ald_bdfe_gxtb_kcal | 38 | 0.0991 | EXPENSIVE (--ohess Hessian+RRHO) |
| prod_bde_gxtb_kcal | 6 | 0.4835 | cheap (SP/opt) |
| ald_bde_gxtb_kcal | 4 | 0.5868 | cheap (SP/opt) |

Summed importance: BDE (cheap, SP/opt only) = 1.0703; BDFE (expensive, full `--ohess` Hessian+RRHO) = 0.2902; ratio BDFE/BDE = 0.27.

**Interpretation**: if BDFE's importance is small relative to BDE's, a cost-aware variant dropping the 2 expensive BDFE columns (keeping only BDE) is worth training and comparing against the full 275-feat champion — for prospective screening of NEW molecules, BDE alone (no Hessian needed) is far cheaper per molecule than BDFE.

## Mordred re-slimming

On top of the existing 199-feat SHAP-pruned mordred set, kept **199/199** after this round's importance+correlation pass (dropped 0 newly-redundant, |corr|>0.9 with a higher-ranked kept feature — note: importance ranking now reflects the *joint* 275-feat model, not the isolated mordred510 model this selection was originally made from, so some previously-kept feats may now look more/less useful).

**Cost-aware candidate**: 273 feats (drops the 2 expensive BDFE cols, keeps BDE) — needs its own retrain to confirm it holds the accuracy; see `mordredslim271_bdegxtb_slim_selection_20260707.json` (`cost_aware_feats_total`).

## Top-25 global importance

| rank | feature | mean|SHAP| | family |
|---|---|---|---|
| 1 | P_int | 1.8489 | QM(72) |
| 2 | ald_wbo_CO | 0.8527 | QM(72) |
| 3 | ald_P_int | 0.6099 | QM(72) |
| 4 | ald_bde_gxtb_kcal | 0.5868 | g-xTB BDE/BDFE |
| 5 | ald_xtb_dipole | 0.5723 | QM(72) |
| 6 | prod_bde_gxtb_kcal | 0.4835 | g-xTB BDE/BDFE |
| 7 | ald_pa_CHO_O | 0.4522 | QM(72) |
| 8 | xtb_dipole | 0.3884 | QM(72) |
| 9 | wbo_CC_new | 0.3311 | QM(72) |
| 10 | vbur_ketC | 0.3039 | QM(72) |
| 11 | mulliken_carbC | 0.2452 | QM(72) |
| 12 | ald_mulliken_CHO_C | 0.2339 | QM(72) |
| 13 | ald_mordred_Mor02se | 0.2279 | mordred |
| 14 | fukui_plus_ketC | 0.2114 | QM(72) |
| 15 | prod_bdfe_gxtb_kcal | 0.1910 | g-xTB BDE/BDFE |
| 16 | mordred_TopoPSA(NO) | 0.1892 | mordred |
| 17 | ald_mordred_PNSA3 | 0.1799 | mordred |
| 18 | mordred_PNSA3 | 0.1742 | mordred |
| 19 | mordred_Mor03v | 0.1707 | mordred |
| 20 | mordred_GeomRadius | 0.1687 | mordred |
| 21 | ald_SASA_total | 0.1668 | QM(72) |
| 22 | ald_mulliken_CHO_O | 0.1662 | QM(72) |
| 23 | ald_mordred_GeomDiameter | 0.1506 | mordred |
| 24 | ald_mordred_TPSA | 0.1490 | mordred |
| 25 | ald_mordred_WNSA3 | 0.1455 | mordred |
