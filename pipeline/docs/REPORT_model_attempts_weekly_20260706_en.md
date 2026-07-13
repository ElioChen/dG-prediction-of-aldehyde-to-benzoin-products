# This Week's Prediction-Model Attempts and Results (2026-07-06)

**Model lineage (headline metric = test MAE of the g-xTB→DFT correction model, kcal/mol):**

| Attempt | Feature set / method | test MAE | Verdict | Report location |
|---|---|---|---|---|
| Starting point (baseline) | ENSEMBLE72 (72 feats) | 1.61 | Previously in use | already existed (06-26) |
| Targeted mordred subset | +438-dim mordred (dispersion/size/shape) | **1.517** | Real improvement | `REPORT_MORDRED510_FINAL_20260703.md` + PNG series 70-79 |
| SHAP pruning | 271 feats (72+199, half the size) | **1.525** | Same accuracy at half the features, former preferred model | `REPORT_MORDREDSLIM271_FINAL_20260703.md` + PNG series 90-99 |
| ADCH/QTAIM, morfeus-9, RDKit-434 | 3 orthogonal descriptor families | flat or worse | **Null result**, ruled out | no standalone report, console log only |
| SELFIES/ECFP/sequence-GRU surrogates | predict ΔG directly from structure | 3.0-3.4 (worse than the 2.92 baseline) | **Null result**, ruled out | `pipeline/train_selfies_surrogate.py`, `train_reaction_repr_compare.py` (no md, console table only) |
| GFN2-level BDE/BDFE | bond dissociation energy/free energy | BDE +0.024 (borderline), BDFE null | BDE marginally real, BDFE ruled out | see memory `bde-descriptor-idea.md` |
| g-xTB-consistent BDE/BDFE (quick-check) | +4 feats, 2-member XGB quick check | 1.612→**1.563** | Real (~4x the noise band) | `REPORT_bdfe_gxtb_full_augment_20260706.md` |
| **★ Full production pipeline (final)** | 275 feats, MLP+3×XGB ensemble | 1.525→**1.503** | **Real but modest, new champion** | `REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md` + PNG series **100-109** |

**All reports/figures live in one shared path:**
`/gpfs/scratch1/shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/`

**Current production champion model bundle:**
`pipeline/models/gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib` (MLP+XGB_d8+XGB_d10 ensemble, uncertainty routing, 275 features)

**Net effect this week:** MAE dropped from 1.61 (two weeks ago) to **1.503** by week's end, a cumulative ~7% reduction. The main contributions came from a *targeted* mordred subset (rather than an undiscriminated dump) and g-xTB-consistent BDE features; SELFIES/ECFP/sequence-model surrogates and pure GFN2-level BDFE were explicitly ruled out.
