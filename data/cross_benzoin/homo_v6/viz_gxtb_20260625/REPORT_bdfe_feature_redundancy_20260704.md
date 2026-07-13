# BDFE redundancy vs the existing mordredslim271 feature set (20260704)

n=214,017 molecules. Correlated aldehyde C-H BDFE against all 124 available existing features (22 aldehyde-QM + 102 mordred kept-set feats).

## Top-15 most correlated existing features

| rank | feature | family | Pearson r | n |
|---|---|---|---|---|
| 1 | xtb_omega | aldehyde QM (72-feat) | -0.497 | 214,016 |
| 2 | xtb_LUMO | aldehyde QM (72-feat) | 0.491 | 214,017 |
| 3 | xtb_EA | aldehyde QM (72-feat) | -0.491 | 214,017 |
| 4 | dual_descriptor_CHO_C | aldehyde QM (72-feat) | 0.436 | 214,014 |
| 5 | xtb_mu | aldehyde QM (72-feat) | 0.426 | 214,016 |
| 6 | fukui_plus_CHO_C | aldehyde QM (72-feat) | 0.420 | 214,016 |
| 7 | xtb_gap | aldehyde QM (72-feat) | 0.391 | 214,017 |
| 8 | mordred_Mor12v | mordred (199-feat) | -0.308 | 214,017 |
| 9 | xtb_eta | aldehyde QM (72-feat) | 0.306 | 214,016 |
| 10 | mordred_Mor14se | mordred (199-feat) | 0.297 | 214,017 |
| 11 | mordred_AMW | mordred (199-feat) | -0.276 | 214,017 |
| 12 | wbo_CO | aldehyde QM (72-feat) | 0.257 | 214,017 |
| 13 | mordred_Mor26p | mordred (199-feat) | 0.236 | 214,017 |
| 14 | mordred_Mor19se | mordred (199-feat) | 0.231 | 214,017 |
| 15 | mordred_Mor31p | mordred (199-feat) | 0.224 | 214,017 |

## Summary

- Strongest single correlation: |r|=0.497 (xtb_omega)
- 9/124 existing features have |r|>0.3 with BDFE
- 0/124 existing features have |r|>0.5 with BDFE

**BDFE is NOT strongly redundant with any single existing feature.** Combined with its weak direct correlation with dG (r~0.04-0.10, see REPORT_aldehyde_bdfe_analysis_20260704.md) but real tree-model MAE gain (+0.024) when added explicitly, the most consistent interpretation is that BDFE carries **distributed, nonlinear/interaction information** that individual existing descriptors don't capture on their own -- plausible given BDFE is a genuinely different physical quantity (a THERMODYNAMIC, ohess-derived free energy of a bond-breaking process) vs. the existing descriptors (static single-point electronic/steric properties of the intact molecule).
