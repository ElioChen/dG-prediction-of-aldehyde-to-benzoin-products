# Conformer label-noise floor (K=5 DFT SP) — 20260626_1559

- molecules attempted: **34**, with >=2 good conformers: **34**
- molecules with <2 conformers (excluded): 0

## Per-molecule ΔG spread across conformers (kcal/mol)
| metric | std (dG_std_kcal) | range (dG_range_kcal) |
|---|---|---|
| mean | 2.677 | 7.400 |
| median | 2.275 | 6.388 |
| 75th pct | 3.407 | 8.261 |
| 90th pct | 4.180 | 12.337 |
| max | 9.802 | 30.131 |

## Implied MAE floor from single-conformer labels
- mean per-mol ΔG std = **2.677** kcal/mol
- proxy MAE contribution (std·√(2/π)) = **2.136** kcal/mol

## Interpretation
The production ensemble test MAE is ~1.6 kcal/mol. The single-conformer label noise contributes ~2.14 kcal/mol (mean) of irreducible error on its own. If this is a large fraction of 1.6, the model is near the data-quality ceiling and further feature/architecture work cannot help — only better labels (Boltzmann-averaged multi-conformer DFT, or a higher-level functional) would move the floor. See [[delta-mae-noise-floor]], [[conformer-search-noise]].
