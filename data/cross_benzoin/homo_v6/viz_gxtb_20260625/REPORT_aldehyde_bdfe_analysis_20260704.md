# Aldehyde C-H BDFE: distribution + dG correlation (20260704)

n=214,048 molecules with valid, physically-sane BDFE (of 218,724 total in the sidecar, 97.9% overall fill rate; 5 pathological outliers with |BDFE|>200 kcal/mol excluded -- almost certainly non-converged-SCF garbage energies, not real chemistry, saved to `ald_bdfe_pathological_outliers_20260704.csv` for follow-up).

## Distribution

- mean=92.88, median=92.22, std=9.85 kcal/mol
- range: -103.2 to 111.4 kcal/mol
- **aromatic** (n=141,492): mean=89.31, std=7.94
- **aliphatic** (n=71,556): mean=99.86, std=9.52

## Correlation with reaction dG

- vs **dG_gxtb** (n=213,984): Pearson r=0.044 (p=4.28e-91), Spearman rho=-0.024
- vs **dG_orca (real DFT)** (n=213,985): Pearson r=-0.103 (p=0.00e+00), Spearman rho=-0.153

## Correlation with g-xTB baseline error

- BDFE vs |g-xTB - DFT| (n=213,982): Pearson r=-0.125 (p=0.00e+00)

## Interpretation

BDFE shows a **weak** linear correlation with the reaction dG (r=0.044 vs g-xTB). This is consistent with it being a genuinely *orthogonal* mechanistic descriptor rather than a redundant restatement of the existing electronic descriptors (wbo_CO etc.) -- if it were fully redundant with them, its raw correlation with dG would likely track much closer to those descriptors' own (typically moderate-to-strong) correlations, and the ~0.024 MAE gain seen when adding it explicitly would be harder to explain.
