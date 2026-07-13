# Full (aldehyde+product) BDFE-augmented feature comparison (20260704)

Both-sides BDFE (v2, DMSO-consistent) test, now that products_bdfe2_descriptors.csv is complete (1435/1463 chunks, 28 timeout, ~1.9% gap accepted). Same 70:20:10 split (seed 42), XGB_d8+XGB_d10 ensemble (quick check, no MLP), vs mordredslim271 preferred production model (test MAE 1.525, full ensemble) and the interim aldehyde-only result (MAE 1.609, delta 0.003 -- noise-level).

- **mordredslim271**: n_feat=271 n=219,095 MAE=1.612 RMSE=2.441 R2=0.854 scope={'aromatic': 1.4299047081578449, 'aliphatic': 1.996139428828925}
- **mordredslim271_plus_bdfe_both**: n_feat=273 n=219,095 MAE=1.605 RMSE=2.434 R2=0.855 scope={'aromatic': 1.4264393495348855, 'aliphatic': 1.981883114935798}
