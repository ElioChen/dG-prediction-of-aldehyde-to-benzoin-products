# RDKit-augmented feature comparison (20260701)

Same 70:20:10 split (seed 42), same ensemble (MLP+XGB8+XGB10), vs champion gxtb-dft-correction-champion (72 feats, full-100%-label test MAE 1.566).

- **baseline_72**: n_feat=72 n=218,797 MAE=1.540 RMSE=2.294 R2=0.868 scope={'aromatic': 1.360393389238737, 'aliphatic': 1.9214164524201456}
- **72_plus_rdkit434**: n_feat=506 n=218,797 MAE=1.548 RMSE=2.297 R2=0.867 scope={'aromatic': 1.3646855968937914, 'aliphatic': 1.9384085879233057}
- **qm_plus_rdkit_no_glob**: n_feat=490 n=218,797 MAE=1.564 RMSE=2.318 R2=0.865 scope={'aromatic': 1.3820187142113425, 'aliphatic': 1.950992231012275}

## Noise band (added 20260702, see REPORT_robustness_baseline72_20260702.md)

baseline_72 test MAE across 5 reshuffled 70:20:10 seeds = **1.571 +/- 0.013** (range 1.555-1.587);
5-fold CV = 1.576 +/- 0.010. Treat any single-split MAE difference **smaller than ~0.02-0.03**
as indistinguishable from seed noise, not a real effect.

- baseline_72 (1.540) vs 72_plus_rdkit434 (1.548): delta = 0.008 -> **within the noise band, flat**.
- baseline_72 (1.540) vs qm_plus_rdkit_no_glob (1.564): delta = 0.024 -> **borderline/at the edge of
  the noise band** (~2x the seed std); read as "likely genuinely worse, but not by a large margin" —
  consistent with "the 16 hand-picked globals are a distilled useful subset, dumping the raw 217x2
  in undiscriminated is net negative" rather than a strong effect.

Conclusion unchanged: RDKit-434 does not clear the noise floor. See [[descriptor-search-exhausted]].
