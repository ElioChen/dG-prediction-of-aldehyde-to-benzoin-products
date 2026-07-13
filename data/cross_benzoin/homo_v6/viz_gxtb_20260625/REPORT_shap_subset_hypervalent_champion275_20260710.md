# Subset-only SHAP on the sulfonyl/P/imine/amide hard tail (20260710)

Tier-1a of the 2026-07-10 external-diagnosis review (Action E). Same champion bundle (`gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib`, test MAE 1.503), no retraining. Hard subset = test-set molecules where either the aldehyde or product SMILES matches sulfonyl/has_P/imine/amide SMARTS (the tags found 11.2x/9.5x/3.6x/3.4x enriched in the worst-15% routed set, see REPORT_deep_error_analysis_champion275_20260707.md). n_hard=4,685 (21.4% of test set), matched-size random background n=4,685.

## Top-25 SHAP importance, hard subset only, vs global/background rank

| rank(hard) | feature | mean\|SHAP\|(hard) | mean\|SHAP\|(background) | rank(background) | rank shift |
|---|---|---|---|---|---|
| 1 | P_int | 2.2283 | 1.7590 | 1 | +0 |
| 2 | ald_wbo_CO | 0.9629 | 0.8127 | 2 | +0 |
| 3 | ald_P_int | 0.8078 | 0.5497 | 3 | +0 |
| 4 | ald_bde_gxtb_kcal | 0.7428 | 0.5418 | 4 | +0 |
| 5 | ald_xtb_dipole | 0.7105 | 0.5301 | 5 | +0 |
| 6 | prod_bde_gxtb_kcal | 0.5998 | 0.4359 | 7 | +1 |
| 7 | ald_pa_CHO_O | 0.5225 | 0.4362 | 6 | -1 |
| 8 | xtb_dipole | 0.4725 | 0.3644 | 8 | +0 |
| 9 | vbur_ketC | 0.3636 | 0.2853 | 10 | +1 |
| 10 | wbo_CC_new | 0.3463 | 0.3274 | 9 | -1 |
| 11 | ald_mordred_Mor02se | 0.2848 | 0.2118 | 14 | +3 |
| 12 | mordred_TopoPSA(NO) | 0.2617 | 0.1696 | 16 | +4 |
| 13 | ald_mulliken_CHO_C | 0.2517 | 0.2280 | 12 | -1 |
| 14 | prod_bdfe_gxtb_kcal | 0.2382 | 0.1745 | 15 | +1 |
| 15 | mordred_PNSA3 | 0.2271 | 0.1548 | 21 | +6 |
| 16 | mulliken_carbC | 0.2269 | 0.2433 | 11 | -5 |
| 17 | ald_SASA_total | 0.2217 | 0.1460 | 22 | +5 |
| 18 | ald_mordred_PNSA3 | 0.2166 | 0.1671 | 17 | -1 |
| 19 | fukui_plus_ketC | 0.2153 | 0.2138 | 13 | -6 |
| 20 | mordred_GeomRadius | 0.2041 | 0.1582 | 20 | +0 |
| 21 | ald_mordred_TPSA | 0.1940 | 0.1358 | 23 | +2 |
| 22 | ald_mordred_WNSA3 | 0.1862 | 0.1318 | 25 | +3 |
| 23 | ald_mordred_Mor02m | 0.1851 | 0.1135 | 32 | +9 |
| 24 | ald_mordred_GeomDiameter | 0.1834 | 0.1348 | 24 | +0 |
| 25 | g_TPSA | 0.1783 | 0.1202 | 27 | +2 |

Positive rank shift = MORE important on the hard subset than on a matched-size random background (model relies on it more/differently for hypervalent cases); negative = less important.

## Top-15 SHAP interaction pairs on the hard subset (n=1200)

Diagonal (main effects) excluded -- these are pure pairwise interaction strengths.

| feature 1 | feature 2 | mean\|interaction\| |
|---|---|---|
| ald_P_int | P_int | 0.3172 |
| prod_bde_gxtb_kcal | ald_bde_gxtb_kcal | 0.0990 |
| P_int | ald_mordred_Mor02se | 0.0947 |
| ald_wbo_CO | P_int | 0.0829 |
| ald_mordred_PNSA3 | mordred_PNSA3 | 0.0775 |
| prod_bdfe_gxtb_kcal | prod_bde_gxtb_kcal | 0.0728 |
| ald_wbo_CO | ald_pa_CHO_O | 0.0703 |
| P_int | vbur_ketC | 0.0649 |
| ald_xtb_dipole | xtb_dipole | 0.0639 |
| vbur_ketC | ald_wbo_CO | 0.0637 |
| ald_mulliken_CHO_C | ald_wbo_CO | 0.0590 |
| ald_pa_CHO_O | vbur_ketC | 0.0523 |
| ald_bde_gxtb_kcal | P_int | 0.0520 |
| ald_xtb_dipole | P_int | 0.0463 |
| ald_bdfe_gxtb_kcal | ald_bde_gxtb_kcal | 0.0450 |

See `130_shap_hard_subset_top25_20260710.png` and `131_shap_interaction_heatmap_hard_20260710.png`.
