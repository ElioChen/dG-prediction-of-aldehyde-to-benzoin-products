# Aldehyde-only BDFE-augmented feature comparison (20260704)

Interim check while product-side BDFE array (24422675) is still running -- only the aldehyde C-H BDFE feature is tested here. Same 70:20:10 split (seed 42), XGB_d8+XGB_d10 ensemble (quick check, no MLP), vs mordredslim271 preferred production model (test MAE 1.525, full ensemble).

- **mordredslim271**: n_feat=271 n=219,095 MAE=1.612 RMSE=2.441 R2=0.854 scope={'aromatic': 1.4299047081578449, 'aliphatic': 1.996139428828925}
- **mordredslim271_plus_ald_bdfe**: n_feat=272 n=219,095 MAE=1.609 RMSE=2.437 R2=0.854 scope={'aromatic': 1.4296799058612604, 'aliphatic': 1.988358285441951}
