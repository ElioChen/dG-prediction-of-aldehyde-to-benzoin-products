# Deep error / noise-band analysis: MORDREDSLIM271_BDEGXTB (20260707)

Test set n=21,910, overall MAE=1.503. Noise-band reference (baseline_72, 5 reshuffled seeds): 1.571 +/- 0.013 kcal/mol (REPORT_robustness_baseline72_20260702.md).

## 1. How much of the tail is real vs noise?

Using a 3-sigma cutoff on the noise band (1.610 kcal/mol): **67.7%** of test molecules have error within what pure label noise could explain; **32.3%** (7,075 molecules) show genuinely elevated error the model is failing to capture, not just label jitter.

## 2. Functional-group enrichment in the worst 15% (routed) set

Ratio = P(tag | routed) / P(tag | background); >1 means over-represented in hard cases. Filtered to tags with >=20 occurrences in the routed set.

| tag | % in routed | % in background | enrichment | n(routed) |
|---|---|---|---|---|
| ald_sulfonyl | 18.8% | 1.7% | 11.24x | 617 |
| prod_sulfonyl | 18.8% | 1.7% | 11.24x | 617 |
| ald_has_P | 2.6% | 0.3% | 9.48x | 87 |
| prod_has_P | 2.6% | 0.3% | 9.48x | 87 |
| prod_imine | 5.5% | 1.5% | 3.60x | 181 |
| ald_imine | 5.5% | 1.5% | 3.60x | 181 |
| ald_amide | 39.3% | 11.5% | 3.43x | 1293 |
| prod_amide | 39.3% | 11.5% | 3.43x | 1293 |
| ald_ester | 16.6% | 9.2% | 1.81x | 546 |
| prod_ester | 16.6% | 9.2% | 1.81x | 546 |
| prod_nitro | 5.4% | 3.6% | 1.51x | 178 |
| ald_nitro | 5.4% | 3.6% | 1.51x | 178 |
| prod_has_B | 0.7% | 0.5% | 1.48x | 23 |
| ald_has_B | 0.7% | 0.5% | 1.48x | 23 |
| ald_nitrile | 4.9% | 3.6% | 1.38x | 162 |

## 3. Structure-space clustering (Morgan FP, k=6)

Unlike the earlier SHAP-attribution-space clustering (baseline_72, 4 heterogeneous attribution patterns), this clusters by actual molecular structure to see if error concentrates on recognizable chemotypes.

| cluster | n | mean error | dominant tags (>50% prevalence) |
|---|---|---|---|
| 0 | 497 | 2.99 | ald_halogen, prod_halogen |
| 1 | 965 | 2.85 | ald_halogen, prod_halogen |
| 2 | 832 | 2.83 | ald_halogen, prod_halogen |
| 3 | 454 | 2.86 | ald_halogen, prod_halogen |
| 4 | 216 | 3.36 | ald_amide, prod_amide |
| 5 | 323 | 3.08 | (none dominant) |

## 4. Scope split (aromatic vs aliphatic) and size dependence

- **aromatic**: n=14,534, MAE=1.327
- **aliphatic**: n=7,270, MAE=1.870

See `115_error_vs_molwt_scope_20260707.png` for the full size x scope error plot, `116_error_vs_uncertainty_20260707.png` for whether the uncertainty-routing signal actually tracks true error, and `117_worst30_tagged_champion275_20260707.png` for the worst-30 molecule structures annotated with functional-group tags.
