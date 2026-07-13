# BDE-augmented feature comparison (20260703)

Same 70:20:10 split (seed 42), XGB_d8+XGB_d10 ensemble (quick check, no MLP), vs mordredslim271 preferred production model (test MAE 1.525, full ensemble).

- **mordredslim271**: n_feat=271 n=219,095 MAE=1.612 RMSE=2.441 R2=0.854 scope={'aromatic': 1.4299047081578449, 'aliphatic': 1.996139428828925}
- **mordredslim271_plus_bde**: n_feat=273 n=219,095 MAE=1.588 RMSE=2.412 R2=0.857 scope={'aromatic': 1.4047173870523284, 'aliphatic': 1.975273433443793}
