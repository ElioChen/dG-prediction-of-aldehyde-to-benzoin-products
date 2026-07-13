# Δ-learning vs. direct DFT regression — diagnostic

**Date:** 2026-06-11 · **Data:** partial run (187/200 ΔG labels reconstructed from
the running ORCA job logs) · **Status:** PRELIMINARY — to be re-run on the full,
clean 200-molecule set. Small-`n` CV metrics are noisy; do not over-read.

## Question

Given that xTB and ORCA use the **same geometry** (ORCA only re-does the
single-point energy; the xTB RRHO thermal corrections cancel), so the learned
correction

```
y = ΔG_orca − ΔG_xtb  =  [E_orca − E_xtb]_electronic   (same geometry)
```

is a smooth electronic-energy difference — should we:

1. try other (neural-network) models,
2. enlarge the dataset,
3. add more features, or
4. **fall back to regressing ΔG_DFT directly** from descriptors (drop Δ-learning)?

## Key data hygiene finding

A single non-physical row (`index=110738`, ΔG_xtb = −160, ΔG_orca = −184 kcal/mol)
**dominated the statistics**. With it removed:

| quantity (clean 187) | value |
|---|---|
| `corr(ΔG_xtb, ΔG_orca)` | **0.58** (moderate, not strong) |
| std ΔG_xtb / ΔG_orca / correction | 5.22 / 5.99 / **5.18** |

So the correction variance (26.8) is only **1.34×** smaller than the DFT variance
(35.9) — Δ-learning reduces the target spread, but modestly. (My earlier "much
easier target" claim came from the −160 outlier inflating the correlation.)

**Physics filter applied:** xTB ΔG must be **negative** for benzoin condensation
(xTB systematically over-stabilises the product). 5 rows with ΔG_xtb ≥ 0 are
suspect failed-xTB cases and were dropped → **182 clean rows**.

## Head-to-head (clean 182, RandomForest, 5×3 repeated-K-fold CV)

| approach | MAE (kcal/mol) | R² |
|---|---|---|
| **Δ-learning** (predict correction, add back ΔG_xtb) | **2.51** | **0.487** |
| Direct DFT (predict ΔG_orca; ΔG_xtb is a feature) | 2.73 | 0.383 |
| xTB baseline (correction ≡ 0) | 11.43 | −4.85 |

➡ **Δ-learning wins** (R² 0.487 vs 0.383). The fallback (direct regression) is
*worse*. **Keep Δ-learning.**

## Learning curve (Δ-learning, 5×3 CV R²)

| train fraction | n | R² | MAE |
|---|---|---|---|
| 40% | 72 | 0.54 | 2.35 |
| 60% | 109 | 0.58 | 2.49 |
| 80% | 145 | 0.47 | 2.60 |
| 100% | 182 | 0.44 | 2.63 |

➡ R² does **not** rise with more data (flat-to-decreasing). So **dataset size is
not the binding constraint** — naively adding same-distribution data won't lift the
fit. (Caveat: small-`n` CV R² is high-variance; treat as indicative.)

## Recommendations (evidence-based)

1. **Other NN — not now.** At ~180 points RF already beats XGBoost/GBT; NNs need
   *more* data than trees and would underperform. Revisit only after a large
   dataset expansion (where the A100 budget becomes useful).
2. **Enlarge dataset — not the bottleneck for fit**, but still worth a modest bump
   (k = 200 → 300–400) for **chemical-space coverage / generalisation** to new
   molecules, which CV R² doesn't directly measure.
3. **More features — targeted, not bulk.** The correction is a pure *electronic*
   energy difference, so features that describe the xTB↔DFT electronic gap (xTB
   energy components, electron-count / size scaling) are the rational additions.
   With 63 features / 182 points the ratio is already thin; add selectively.
4. **Direct DFT regression — no**, it underperforms Δ-learning here.

## Reframe — don't chase R²

The correction has a narrow dynamic range (std ≈ 5 kcal/mol), which makes R²
pessimistic. For a ΔG property the meaningful metric is **MAE ≈ 2.5 kcal/mol**,
which is already useful for **screening-level** prediction (chemical accuracy
≈ 1 kcal/mol). The model is usable; the low R² is partly an artefact of the small
target range.

## Next step

Re-run this exact diagnostic on the **full clean 200** (after the supplementary
ORCA job fills the 9 timed-out molecules) before committing to feature work or a
subset expansion.
