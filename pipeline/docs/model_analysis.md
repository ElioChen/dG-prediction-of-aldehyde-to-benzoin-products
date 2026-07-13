# Deep analysis of the Δ-learning model — where the error comes from & how to improve

**Date:** 2026-06-12 · **Model:** GBT, 472 clean molecules, 63 features ·
**CV:** MAE 3.24, R² 0.627 (xTB baseline MAE 10.14).

## 1. The error is dominated by CONFORMER NOISE, not the model

Out-of-fold |error| rises monotonically with molecular flexibility:

| RotBonds | n | MAE |
|---------:|--:|----:|
| 0–2 (rigid)   | 134 | **2.47** |
| 3–5           | 186 | 2.82 |
| 6–9           | 124 | **4.23** |
| 10+ (flexible)| 28  | **5.31** |

The learned correction is a pure electronic ΔΔE on a *single* geometry. For
flexible molecules the conformer xTB picks as "best" is often not best at DFT
level, so the ORCA single-point sits on a sub-optimal geometry → the **ΔG label
itself is noisy**. This is a data-quality ceiling, not a model deficiency.

## 2. Data QUANTITY is not the lever

Learning curve (GBT, repeated CV) plateaus by ~280 molecules:

| n | MAE | R² |
|--:|----:|---:|
| 188 | 3.71 | 0.54 |
| 283 | 3.07 | 0.66 |
| 377 | 3.22 | 0.64 |
| 472 | 3.22 | 0.63 |

More same-quality data won't move the fit. (Diverse data already added; coverage
of the 218k library is intrinsically unreachable — see expansion doc.)

## 3. Model choice is marginal — all families hit the same ceiling

xgb / rf / gbt all land at CV MAE ≈ 3.0–3.2 on this data. The 120-trial SLURM
sweep refines this but won't break the conformer-noise ceiling. Robust losses
(Huber) are worth a try since the noise is heteroscedastic in RotBonds.

## 4. Features are balanced — no cheap pruning win

Mean |SHAP| by group: **xTB 20%, Multiwfn (ADCH/QTAIM) 19%, RDKit/Gasteiger ~61%**.
Top features: `gasteiger_CHO_O`, `dG_xtb_kcal`, `adch_fukui_minus_CHO_O` (Multiwfn,
#3), `xtb_mu`, `xtb_eta`, `HBA`, `xtb_dipole`, `LogP`. Multiwfn pulls ~1/5 of the
signal — keep it for the high-accuracy package (the RDKit-only browser surrogate
will be measurably weaker).

## Improvement plan (ranked by expected impact)

1. **Conformer ensembles for the ΔG labels (biggest lever).** Re-compute the
   ~150 RotBonds≥6 molecules with Boltzmann-weighted ORCA single-points over the
   top-N xTB conformers (or DFT geometry opt). Budget is ample. Expected: flexible
   MAE ~4.5→~2.5, overall ~3.2→~2.5–2.8. This is a data-gen change to the thermo
   pipeline + a targeted SLURM re-run.
2. **Robust-loss / thorough model sweep** (running, job 23709558). Marginal.
3. **Sample weighting / uncertainty** by RotBonds — report per-prediction
   uncertainty rather than chase a single MAE.
4. **Not worth it:** more same-quality data (plateaued); dropping Multiwfn (loses 19%).

## Honest performance statement

MAE ≈ 3.2 kcal/mol overall, **≈ 2.5 for rigid aldehydes**, degrading to ~5 for very
flexible ones. Useful for screening; the AD flag + (future) RotBonds-uncertainty
tell the user when to trust a prediction.
