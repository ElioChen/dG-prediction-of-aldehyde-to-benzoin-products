# Aromatic-subset deep dive: MORDREDSLIM271_BDEGXTB (20260707)

## 1. Is the old MaxMin sampling bias still a concern?

Full-library population (`aldehyde_class.parquet`, n=220,724): aromatic 66.3%, aliphatic 33.2%. Test-set population (n=21,910): aromatic 66.3%, aliphatic 33.2%. Gap = 0.0 points. **Negligible gap — the earlier MaxMin active-selection bias (39%->11% undersampling) is NOT present in the current training population**, since labeling now covers the near-full library rather than a MaxMin-selected subset, and the 70:20:10 split is a uniform random permutation with no scope-based stratification needed.

## 2. Per-scope test accuracy

| scope | n_test | MAE | bias (pred-true) | R2 | full-lib route_to_dft rate |
|---|---|---|---|---|---|
| aromatic | 14,534 | 1.327 | +0.002 | 0.875 | 9.7% |
| aliphatic | 7,270 | 1.870 | +0.104 | 0.842 | 26.5% |

## 3. Within aromatic: heteroaromatic vs simple carbocyclic

| chemotype | n | MAE | mean bias |
|---|---|---|---|
| simple_carbo_aromatic | 6714 | 1.268 | +0.033 |
| azine_or_azole_N | 5835 | 1.391 | -0.033 |
| furan | 563 | 1.375 | -0.052 |
| thiophene+azine_or_azole_N | 554 | 1.432 | +0.057 |
| furan+azine_or_azole_N | 430 | 1.222 | -0.103 |
| thiophene | 405 | 1.275 | +0.066 |
| furan+thiophene+azine_or_azole_N | 30 | 1.455 | +0.238 |
| furan+thiophene | 3 | 3.886 | +3.133 |

See `119_parity_aromatic_champion275_20260707.png` / `119_parity_aliphatic_champion275_20260707.png` for scope-split parity, `120_aromatic_error_vs_size_20260707.png` for the chemotype x size breakdown within the aromatic subset, and `118_aromatic_hetero_mae_20260707.png` for the chemotype MAE bar chart.
