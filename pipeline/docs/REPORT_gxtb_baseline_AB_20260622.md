# g-xTB vs GFN2 as the Δ-learning baseline — A/B result (2026-06-22)

Tests recommendation B of `REPORT_conformer_xtb_dft_synthesis_20260622.md`: replace the
Δ-model's semiempirical baseline (GFN2 `dG_xtb_kcal`) with g-xTB, evaluated on the SAME
funnel_v3 geometry the DFT label sits on, all-solvated (GFN2 ALPB-DMSO, g-xTB COSMO-DMSO,
DFT CPCM-DMSO). Baseline computed by `pipeline/compute/gxtb_baseline.py` (SLURM 24101761,
1695/1695 in-scope mols). A/B by `pipeline/gxtb_ab.py` on matched rows (same molecules,
same features/model/CV; only the baseline changes; corr-MAD QC off for a fair match).

## Result (matched in-scope n=1663)

| baseline | raw MAE | raw bias | raw r | CV MAE | CV RMSE | CV R² |
|---|---|---|---|---|---|---|
| GFN2  | 12.95 | −12.94 | 0.503 | 2.52 | 3.88 | 0.437 |
| g-xTB |  3.03 | −0.64  | 0.709 | 2.10 | 3.06 | 0.650 |

Δ(CV MAE) g-xTB − GFN2 = **−0.42 kcal/mol**; raw baseline MAE **13.0 → 3.0**.

## Reading

1. **Raw baseline, in DMSO, on the training set:** g-xTB cuts MAE 4.3× (13.0→3.0) and
   nearly removes the systematic over-exergonic bias (−12.9 → −0.6). Confirms the 1%
   validation finding at scale, now fully solvated.
2. **Δ-model CV floor drops 2.52 → 2.10** (−0.42), RMSE 3.88→3.06, R² 0.44→0.65. So the
   floor was NOT purely irreducible conformer-label noise — a better physical baseline
   lowers it. The GFN2 baseline left structured error the ML only partly recovered.
3. **Robust to outliers:** g-xTB with NO outlier-QC (2.10) already beats GFN2 even with the
   production corr-MAD QC (~2.2, per delta-mae-noise-floor) — g-xTB does not produce the
   catastrophic over-exergonic outliers GFN2 needs QC to suppress.
4. **Biggest win is extrapolation:** a 4× better raw baseline means the 220k-library screen
   relies far less on the ML correction outside the training AD.

NB the GFN2 CV here (2.52) is higher than the shipped ~2.2 because this A/B turns OFF the
corr-MAD QC for a fair matched comparison; the apples-to-apples contrast is 2.52 vs 2.10.

## Figure
`figs/baseline_gxtb_vs_gfn2_controlled_20260622.png` — controlled scatter (same funnel_v3
geometry, both DMSO), superseding the mixed-condition `semiempirical_vs_dft_dG_20260622.png`.

## Recommendation
Switch the production Δ-model baseline to g-xTB (COSMO-DMSO SP on the funnel_v3 geometry).
Re-fit + re-ship with the corr-MAD QC restored for the final number. Cost is pure
semiempirical (~280 core-h for the full label set). Route the residual EWG endergonic tail
to DFT as before ([[no-dG-extreme-filtering]]).
