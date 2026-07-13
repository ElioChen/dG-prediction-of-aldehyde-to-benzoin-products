# BDE/BDFE surrogate model (20260704)

Predicts per-molecule bond dissociation (free) energy from the existing cheap descriptors (72-champion QM + mordredslim271 kept mordred subset), instead of running real xtb (--ohess for BDFE). Same 70:20:10 split (seed 42), XGB_d8+XGB_d10 ensemble + quantile(05/95) UQ, per target.

## Results

| target | n | n_feat | test MAE | RMSE | R2 |
|---|---|---|---|---|---|
| bde_ald_CH_kcal (ald_bde) | 214,954 | 140 | 3.027 | 9.700 | 0.094 |
| bdfe_xtb_kcal (ald_bdfe) | 213,191 | 140 | 2.018 | 4.436 | 0.797 |
| bde_prod_CC_kcal (prod_bde) | 208,907 | 147 | 3.878 | 10.334 | 0.652 |
| bdfe_xtb_kcal (prod_bdfe) | 205,339 | 147 | 2.704 | 5.117 | 0.901 |

## Interpretation

- **ald_bde**: R2=0.094 -> redundancy is **LOW (carries genuinely independent information)**. Top predictive feats: mordred_Mor17, g_nAliphRing, mordred_Mor07m, mordred_Mor23p, mordred_Mor20v
- **ald_bdfe**: R2=0.797 -> redundancy is **MODERATE**. Top predictive feats: xtb_LUMO, wbo_CO, fukui_plus_CHO_C, xtb_EA, dual_descriptor_CHO_C
- **prod_bde**: R2=0.652 -> redundancy is **MODERATE**. Top predictive feats: mordred_MOMI-Y, SASA_total, mordred_PBF, sterimol_B5, mordred_Mor30m
- **prod_bdfe**: R2=0.901 -> redundancy is **HIGH (mostly redundant with existing feats)**. Top predictive feats: wbo_CC_new, wbo_CO_ket, xtb_LUMO, hb_angle, g_nAromRing

If R2 is high enough for practical use (rule of thumb R2>0.85, MAE well under the real xtb run's own noise), the corresponding bundle can be used as a fast prospective-screening substitute for real xtb BDE/BDFE on new molecules outside the current 220k library.

## Model bundles

- `/scratch-shared/schen3/benzoin-dg/pipeline/models/bde_surrogate_ald_bde_20260704.joblib`
- `/scratch-shared/schen3/benzoin-dg/pipeline/models/bde_surrogate_ald_bdfe_20260704.joblib`
- `/scratch-shared/schen3/benzoin-dg/pipeline/models/bde_surrogate_prod_bde_20260704.joblib`
- `/scratch-shared/schen3/benzoin-dg/pipeline/models/bde_surrogate_prod_bdfe_20260704.joblib`
