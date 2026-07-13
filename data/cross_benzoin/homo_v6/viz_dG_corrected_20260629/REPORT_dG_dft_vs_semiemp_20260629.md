# Raw ΔG correlation: DFT vs GFN2 / g-xTB  (20260629)
DFT = r2SCAN-3c (`dG_orca_kcal`, n DFT labels matched = 214,260; usable 213,179, dropped 70 with |DFT| ≥ 60.0).

| method (x) | Pearson R | R² | MAE | RMSE | bias(method−DFT) | slope | intercept |
|---|---|---|---|---|---|---|---|
| GFN2-xTB | 0.584 | 0.341 | 15.63 | 16.44 | -15.61 | 0.85 | +14.17 |
| g-xTB | 0.599 | 0.359 | 4.15 | 5.77 | -2.41 | 0.73 | +3.32 |

## Interpretation
- **g-xTB tracks DFT far better than GFN2**: R²=0.359 vs 0.341, MAE=4.15 vs 15.63 kcal/mol. g-xTB is the right Δ-baseline; GFN2 is not.
- Bias: GFN2 -15.61, g-xTB -2.41 kcal/mol (method − DFT). A slope ≠ 1 means the error is partly systematic (correctable), motivating the ML Δ-correction.
## Figures
- `08_parity_DFT_vs_gfn2.png`
- `09_parity_DFT_vs_gxtb.png`
