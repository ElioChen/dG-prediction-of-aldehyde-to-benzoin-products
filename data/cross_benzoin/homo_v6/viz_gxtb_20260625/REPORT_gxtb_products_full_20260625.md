# Full-library g-xTB product-descriptor analysis (20260625)

**Source**: `/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/products_all.csv` — job 24128375, drained 2026-06-24 ~06:02.

## 1. Coverage & data quality

- Rows: **220,724** products (all `reaction_type=homo`, all `is_homo=True`).
- xTB-optimized: **219,680** (100.0%); error rows: 1,303.
- Both ΔG present (GFN2 & g-xTB): **219,418** (99.4%).
- Robust core |ΔG|<60 for both: **219,323** (99.96% of valid); 95 pathological-|ΔG| pairs excluded from parity stats.
- **Empty descriptor families (16 cols, MULTIWFN=0)**: all ADCH + QTAIM are 100% NaN -> unusable here:
  `adch_ketC, adch_ketO, adch_carbC, adch_hydO, adch_hydH, adch_fukui_plus_ketC, adch_fukui_minus_ketC, adch_fukui_plus_carbC, adch_fukui_minus_carbC, qtaim_rho_CO_ket, qtaim_lap_CO_ket, qtaim_ell_CO_ket, qtaim_rho_CC_new, qtaim_lap_CC_new, qtaim_ell_CC_new, qtaim_rho_HB`

## 2. g-xTB vs GFN2-xTB agreement (the baseline question)

- Pearson r = **0.437**, bias (g-xTB−GFN2) = **+13.12**, MAE = **13.24**, RMSE = **14.09** kcal/mol (core, n=219,323).
- GFN2 mean ΔG **-9.73** (exergonic) vs g-xTB **3.38** -> the two methods disagree on reaction *sign* on average: a **+13.12 kcal/mol systematic shift**, not just scatter.

## 3. What drives the disagreement

Top |r| descriptors vs (g-xTB−GFN2) residual:

- `xtb_IP`: r = +0.360
- `xtb_HOMO`: r = -0.345
- `xtb_mu`: r = -0.338
- `xtb_LUMO`: r = -0.313
- `P_int`: r = -0.258
- `xtb_omega`: r = +0.253
- `fukui_minus_ketC`: r = +0.246
- `dual_carbC`: r = -0.205

## 4. Pathological ΔG tail

- **95** pairs have |ΔG|>60 in at least one method (0.04%) — likely broken SCF / strained geometry, candidates to route to DFT (never delete; see no-ΔG-extreme-filtering).
- of those, g-xTB-only extreme: 28; GFN2-only extreme: 23; both: 44.

## 5. Figures

- `01_descriptor_family_coverage.png`
- `02_dG_distributions.png`
- `03_parity_gxtb_vs_gfn2.png`
- `04_residual_distribution.png`
- `05_residual_descriptor_correlations.png`
- `06_residual_vs_top_driver.png`
- `07_dG_descriptor_correlations.png`
- `08_correlation_heatmap.png`
- `09_homolumo_gap.png`
- `10_omega_vs_dG.png`
- `11_fukui_dual_ketC.png`
- `12_vbur_ket_vs_carb.png`
- `13_hbond_geometry.png`

## 6. Takeaways

1. **g-xTB is not a drop-in for GFN2 on ΔG**: r≈0.44 but a **+13.1 kcal/mol systematic bias** flips the average reaction from exergonic (GFN2) to endergonic (g-xTB). Calibrate before use as a Δ-baseline.
2. The disagreement is **structured, not random** — it correlates with the descriptors in fig 05, so a small linear/affine correction keyed on those should collapse most of the bias.
3. **ADCH+QTAIM are absent full-library** (MULTIWFN=0). If those features matter for the surrogate, they must be back-filled on a sampled subset (see multiwfn-env-and-screen-gap).