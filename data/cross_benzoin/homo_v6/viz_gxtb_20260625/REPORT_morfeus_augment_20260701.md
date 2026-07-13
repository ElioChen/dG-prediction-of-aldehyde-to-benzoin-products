# Morfeus-augmented feature comparison (20260701)

Same 70:20:10 split (seed 42), same ensemble (MLP+XGB8+XGB10), vs champion gxtb-dft-correction-champion (72 feats, full-100%-label test MAE 1.566).

- **baseline_72**: n_feat=72 n=219,095 MAE=1.574 RMSE=2.387 R2=0.860 scope={'aromatic': 1.3801811197079614, 'aliphatic': 1.9780853713515691}
- **72_plus_morfeus9**: n_feat=81 n=219,095 MAE=1.576 RMSE=2.393 R2=0.860 scope={'aromatic': 1.3864870357245476, 'aliphatic': 1.9725202123499772}

## Noise band (added 20260702, see REPORT_robustness_baseline72_20260702.md)

baseline_72 test MAE across 5 reshuffled 70:20:10 seeds = **1.571 +/- 0.013** (range 1.555-1.587);
5-fold CV = 1.576 +/- 0.010. This run's own baseline_72 (1.574) already sits inside that band, so
the +0.002 delta to 72_plus_morfeus9 (1.576) is **far inside noise — no detectable effect either
way**, not even a borderline case like the RDKit-no-glob variant.

Conclusion unchanged: morfeus-9 does not clear the noise floor. See [[descriptor-search-exhausted]].
