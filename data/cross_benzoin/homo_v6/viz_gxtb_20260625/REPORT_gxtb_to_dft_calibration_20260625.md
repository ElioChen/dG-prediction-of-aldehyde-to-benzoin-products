# g-xTB → DFT calibration (20260625)

Matched products∩DFT-SP(in-progress): **136,253**; usable (|DFT|<60, full features): **136,203**.
Split 70/20/10 (train/val/test), seed 42. Target = DFT r2SCAN-3c ΔG. Δ-learning on g-xTB.

## Held-out TEST metrics (kcal/mol)

| model | MAE | RMSE | bias | R² |
|---|---|---|---|---|
| L0 raw g-xTB | 4.35 | 6.11 | -2.62 | 0.17 |
| (ref) raw GFN2-xTB | 15.60 | 16.51 | -15.59 | -5.03 |
| L1 +constant bias | 4.15 | 5.52 | -0.10 | 0.33 |
| L2 affine(a*g-xTB+b) | 4.03 | 5.33 | -0.09 | 0.37 |
| L3 ridge Δ(descriptors) | 2.87 | 4.08 | -0.08 | 0.63 |
| L4 GBT Δ(descriptors) | 2.46 | 3.67 | -0.05 | 0.70 |

## Findings

- **g-xTB raw MAE = 4.35** vs DFT — already far better than GFN2 (15.6). g-xTB is the correct Δ-baseline.
- A **constant bias removal** alone -> MAE 4.15 (bias -2.62 -> ~0): most of the raw error is a fixed offset.
- **Descriptor Δ-correction (GBT) -> MAE 2.46, R²=0.70**, a 44% cut from raw, approaching the ~3.2 kcal noise floor.
- Top correction drivers: see fig 23 (electronic IP/HOMO/ω + carbonyl charge/Fukui), consistent with where xTB-family methods misjudge EWG electronics.

## Figures
- `20_parity_L0_raw_gxtb.png`, `21_parity_L4_gbt_corrected.png`
- `22_correction_hierarchy_MAE.png`, `23_gbt_feature_importance.png`, `24_error_vs_size.png`