# g-xTB-consistent BDFE/BDE augmented feature comparison (20260706)

Method-mismatch hypothesis test: GFN2-level BDFE was a null result (finalize_correction_bdfe_full.py, +0.007 MAE). This uses g-xTB-consistent BDFE/BDE (calc_bde_free_energy_gxtb.py), full library both sides (aldehydes 1471/1471, products 1460/1463, 3 timeout gap accepted). Same 70:20:10 split (seed 42), XGB_d8+XGB_d10 ensemble, vs mordredslim271 preferred production model (test MAE 1.525).

- **mordredslim271**: n_feat=271 n=219,095 MAE=1.612 RMSE=2.441 R2=0.854 scope={'aromatic': 1.4299047081578449, 'aliphatic': 1.996139428828925}
- **mordredslim271_plus_bdfe_gxtb_both**: n_feat=273 n=219,095 MAE=1.605 RMSE=2.434 R2=0.855 scope={'aromatic': 1.4222316028794069, 'aliphatic': 1.9900246966964714}
- **mordredslim271_plus_bde_gxtb_both**: n_feat=273 n=219,095 MAE=1.580 RMSE=2.396 R2=0.859 scope={'aromatic': 1.3928032994438475, 'aliphatic': 1.9740938403015558}
- **mordredslim271_plus_bdfe_and_bde_gxtb_both**: n_feat=275 n=219,095 MAE=1.563 RMSE=2.374 R2=0.862 scope={'aromatic': 1.378252871856894, 'aliphatic': 1.9525746522833745}
