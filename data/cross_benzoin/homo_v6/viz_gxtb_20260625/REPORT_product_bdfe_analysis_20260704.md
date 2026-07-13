# Product C-C BDFE: distribution + dG correlation (20260704)

n=207,828 molecules with valid, physically-sane BDFE (of 215,367 total in the sidecar, 96.0% overall fill rate; 4 pathological outliers with |BDFE|>200 kcal/mol excluded).

## Distribution

- mean=60.84, median=56.01, std=16.49 kcal/mol
- range: -141.7 to 117.7 kcal/mol
- **aromatic** (n=137,320): mean=51.44, std=9.22
- **aliphatic** (n=69,521): mean=79.25, std=11.26

**Compared to aldehyde C-H BDFE** (n=214,017): mean=92.86, std=9.86 kcal/mol -- product C-C bond is WEAKER on average (delta=-32.0 kcal/mol).

## Correlation with reaction dG

- vs **dG_gxtb** (n=207,825): Pearson r=-0.173 (p=0.00e+00), Spearman rho=-0.164
- vs **dG_orca (real DFT)** (n=207,820): Pearson r=-0.262 (p=0.00e+00), Spearman rho=-0.254

## Correlation with g-xTB baseline error

- BDFE vs |g-xTB - DFT| (n=207,817): Pearson r=-0.045 (p=1.73e-92)

## Interpretation

Product C-C BDFE shows a **weak** linear correlation with the reaction dG (r=-0.173 vs g-xTB) -- consistent with the aldehyde-side finding. Note this bond is mechanistically the MOST relevant one (it's the bond literally formed in the reaction), yet the raw correlation is still weak, which fits the earlier finding that BDFE (either side) only gave a noise-level MAE gain (see REPORT_bdfe_full_augment_20260704.md and bde-descriptor-idea memory) when added explicitly to the correction model. Whether descriptors can PREDICT this BDFE well (a separate, nonlinear-reconstruction question) is answered by the surrogate model (see REPORT_bde_surrogate_*.md, prod_bdfe target).
