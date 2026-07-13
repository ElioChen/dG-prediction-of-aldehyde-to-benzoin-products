# Final corrected reaction dG — homo_v6 full library (20260629)
Source: `products_dG_corrected_FINAL_20260626.csv`  ·  n = **218,227** products
## Were dG plots drawn before?
Yes for the **raw** descriptor table (`viz_gxtb_20260625/02_dG_distributions.png`, `07_dG_descriptor_correlations.png`, `10_omega_vs_dG.png`, Jun 25), but **not** for this Jun-26 ML-corrected deliverable with uncertainty + DFT-routing. These 7 figures fill that gap.
## Distribution summary (min / q25 / median / mean / q75 / max)
- raw g-xTB ΔG: -213.61 / 0.45 / 3.52 / 3.37 / 6.17 / 155.89
- **corrected ΔG**: -212.63 / 1.93 / 5.44 / 5.86 / 9.16 / 157.64
- correction (cor−raw): -42.24 / -0.67 / 1.63 / 2.49 / 5.06 / 45.39
- PI-width (uncertainty): 0.36 / 4.51 / 6.45 / 7.05 / 9.06 / 44.41
## Interpretation
- The ML correction shifts the median ΔG **3.52 → 5.44** kcal/mol (median correction **+1.63**): g-xTB systematically under-estimates the endergonicity of homo-benzoin coupling, and the DFT correction is almost entirely a positive (less favorable) shift.
- **15.3%** of products are exergonic at the corrected level (ΔG < 0) vs 22.0% raw — correction prunes the optimistic exergonic tail.
- Routing: **32,774 (15%)** flagged `route_to_dft` (PI width ≥ 10.57). High uncertainty concentrates in the ΔG extremes (Fig 5/6); the confident core is the near-thermoneutral bulk.
- Off-axis extremes (141 below -20, 878 above 30 kcal/mol) are the known EWG / strained / generator-edge cases — kept, not trimmed, per the no-ΔG-extreme-filtering rule; most fall in route_to_dft.
## Figures
- `01_dG_corrected_distribution.png`
- `02_dG_raw_vs_corrected.png`
- `03_correction_magnitude.png`
- `04_uncertainty_pi_width.png`
- `05_dG_vs_uncertainty_hexbin.png`
- `06_dG_by_routing.png`
- `07_cumulative_exergonic_fraction.png`
