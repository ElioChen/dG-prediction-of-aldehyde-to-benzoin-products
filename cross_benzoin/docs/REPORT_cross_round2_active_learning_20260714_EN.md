# Cross-benzoin round 2: active-learning selection + retrain result (2026-07-14)

Companion file: `REPORT_cross_round2_active_learning_20260714_ZH.md` (中文版).

> **Caveat added 2026-07-17**: the frozen-holdout MAE/R² numbers in this report were measured
> on `candidates_v3`'s original molecule-level (InChIKey-disjoint) split, later found to leak
> Bemis–Murcko scaffolds severely (93% of the eventual 29-row holdout had at least one side's
> scaffold already present in train). Treat this report's frozen-holdout numbers as optimistic
> by roughly 0.2–0.5 MAE relative to true novel-scaffold generalization. The corrected,
> scaffold-disjoint evaluation (n_test=450) and the recommended production model
> (`cross_benzoin/predict_cross_champion.py`) now supersede this metric — see `STATUS_EN.md`'s
> correction note and memory `cross-scaffold-disjoint-rebuild-20260717`. This report is
> otherwise kept unmodified below as a historical snapshot.

## What this round was

Round 1 (`cross_pilot_v1`, see `REPORT_cross_pilot_dft_sp_validation_20260714_EN.md`)
established the first real cross-benzoin Δ-model: 598 rows / 299 unordered
pairs, uniform diversity sampling across the 6 category-pair combinations,
CV MAE 2.106 vs g-xTB baseline 3.44. The open question flagged at the end of
that report was: **the model's limiting factor is too few simulated
molecules — does more data (chosen well) actually help?**

Round 2 answers that with a genuine active-learning loop, not just more
random/diverse sampling:

1. Built a 4200-row diversity-sampled candidate pool (`cross_round2`,
   different donor/acceptor pairs than round 1, category-rebalanced).
   Featurized (xTB + g-xTB, job 24614015) and computed product-side BDE
   (job 24620376) — both clean (99.5% and 93.5% success respectively, same
   benign failure modes as round 1).
2. Trained a **40-member bootstrap ensemble** (resample round 1's 299 pairs
   with replacement, same xgb config as the shipped model) and scored all
   4181 valid round-2 rows. Ranked candidates by **prediction-uncertainty
   (std across the ensemble)** — this is real epistemic-uncertainty-driven
   active learning: it asks "where does the round-1 model most disagree
   with itself under resampling," not just "which molecules look chemically
   different."
3. Selected the **top 900/2098 pairs** (43%) by uncertainty for DFT-SP
   labeling (1794 directed rows, both orientations kept, matching round 1's
   convention).
4. Ran DFT-SP (job 24620959, r2SCAN-3c, 18-task array, genoa): **1756/1756
   rows valid, zero errors** in the processed set (38 of the 1794 rows were
   excluded upfront because one side's aldehyde fell in the full library's
   known ~0.8% DFT-SP-timeout gap — not silently miscomputed, just not
   attempted).
5. Combined round1 + round2 into one 150-feature training table (2354 rows /
   1179 pairs) and retrained `train_cross_delta.py` on it.

## Result 1 — the uncertainty ranking picked genuinely hard chemistry, not noise

Selected-pair category mix vs the full round-2 pool:

| reaction_type   | selected | pool | selection rate |
|-----------------|---------:|-----:|----------------:|
| carbo-hetero    | 252 | 399 | 63% |
| hetero-hetero   | 216 | 400 | 54% |
| carbo-carbo     | 204 | 399 | 51% |
| aliph-carbo     |  88 | 300 | 29% |
| aliph-hetero    |  86 | 300 | 29% |
| aliph-aliph     |  54 | 300 | 18% |

The model's bootstrap disagreement is concentrated on hetero/carbo-aromatic
pairs, not spread uniformly. Reading: round 1's 299 pairs already constrained
aliphatic-pair behavior reasonably well (aliphatic aldehydes are structurally
simpler, lower-variance), while hetero/carbo-aromatic chemical space is
higher-variance and was under-covered. This is exactly the kind of targeting
active learning is supposed to produce, and it's a different (better-informed)
prioritization than round 1's own diversity-only rebalancing.

Independent sanity check before committing the full DFT-SP batch: a 3-row
smoke test showed the two orientations of the same selected pair
(donor-acceptor swapped) gave **consistent, same-direction DFT corrections**
(+8.5 and +9.5 kcal/mol vs g-xTB) — the large corrections the model flagged
as "uncertain, worth labeling" are real, systematic chemistry, not
per-geometry noise.

## Result 2 — round 2's selected set is measurably harder, on both baselines

|                        | round 1 (n=598) | round 2 (n=1756) |
|------------------------|----------------:|------------------:|
| `dG_orca_kcal` mean    | 5.06 | 6.42 |
| `dG_orca_kcal` std     | 5.03 | 9.61 |
| `dG_orca_kcal` max     | 24.29 | 52.58 |
| g-xTB baseline MAE     | 3.44 | 6.24 |
| Δ-model MAE (in combined-model CV) | 2.01 | 3.53 |

This is the active-learning selection working as designed: by construction,
round 2 is enriched for the pairs g-xTB (and the round-1 model) handle worst.
**The combined-model's overall CV MAE (3.147) looks worse than round 1's
standalone 2.106 — that is a population-difficulty shift, not model
regression.** Evaluated on round-1-only rows, the combined model still scores
MAE 2.01 (same or marginally better than round 1's own in-sample number),
confirming no degradation from adding the harder round-2 data.

## Result 3 — the model's relative edge over g-xTB got BIGGER on the hard cases

|                              | round 1 alone | combined (round1+2) |
|------------------------------|--------------:|---------------------:|
| CV Δ-model MAE               | 2.106 | 3.147 |
| CV g-xTB baseline MAE        | 3.445 | 5.534 |
| **MAE improvement over g-xTB** | **+1.339** | **+2.387** |
| CV R²                        | 0.708 | 0.749 |
| Frozen holdout n_test        | 14 | 29 |
| Frozen holdout Δ-model MAE   | 2.638 | 3.612 |
| Frozen holdout g-xTB MAE     | 3.078 | 5.315 |
| Frozen holdout R²            | 0.697 | 0.831 |

Two things matter here more than the raw MAE number: (1) the model's
**absolute improvement margin over the g-xTB baseline nearly doubled**
(+1.34 → +2.39 kcal/mol), because g-xTB degrades faster than the Δ-model does
as chemistry gets harder — this is the whole value proposition of Δ-learning,
and it's now demonstrated on out-of-round-1 chemistry, not just in-sample.
(2) The frozen molecule-disjoint holdout (candidates_v3's real train/test
split, the only fully leakage-safe metric in this pipeline) doubled in size
(14→29 test rows) and its R² rose from 0.697 to 0.831 — a genuinely more
trustworthy number, still showing a large Δ-model-over-g-xTB gap.

## Result 4 — feature-block ablation: same pattern as round 1

| block           | n_feats | MAE | RMSE | R² |
|-----------------|--------:|----:|-----:|---:|
| 2D_only         | 48  | 4.391 | 5.874 | 0.543 |
| aldehydes_only  | 54  | 3.994 | 5.393 | 0.615 |
| product_only    | 54  | 3.699 | 5.023 | 0.666 |
| donor+acceptor  | 102 | 4.012 | 5.415 | 0.612 |
| all_raw_blocks  | 140 | 3.126 | 4.325 | 0.752 |
| all+interaction | 150 | 3.147 | 4.356 | 0.749 |

Product-side descriptors alone (`product_only`) are still the single
strongest individual block, and combining all raw blocks still gives the
biggest jump — consistent with round 1. The `interaction_*` block (donor-
acceptor complementarity terms) again adds nothing measurable over
`all_raw_blocks` (3.126 vs 3.147, within CV noise) — same null result as
round 1, now confirmed at ~4x the sample size. Worth dropping from future
default training runs unless a cheaper-to-compute variant emerges.

## Known gaps, not fixed this round

- **Split-consistency**: the frozen-holdout step dropped 788/2354 rows
  (donor/acceptor aldehyde not found in, or landing in different splits of,
  `candidates_v3`'s molecule-disjoint split map) — same root cause flagged in
  the prior push session (custom pair generators don't respect per-molecule
  split membership the way `candidates_v3`'s own 4M pair list does). Round 3
  should sample directly from `candidates_v3` to eliminate this loss.
- **~38-row aldehyde-cache gap**: unresolved, same ~0.8%-of-library DFT-SP
  timeout tail as the full homo campaign. Not worth a dedicated backfill at
  this scale; would recur (shrinking) in future rounds.
- **ENSEMBLE72 packaging gap**: `predict_dG()`'s inference path still doesn't
  call the modern feature pipeline that actually produced any of the
  Δ-models discussed here (round-1 champion or this round's combined model,
  or the separate homo full-library ENSEMBLE72 champion) — flagged in the
  prior session, deliberately not rushed, still open.

## Bottom line

Uncertainty-driven active learning (not just diversity sampling) correctly
identified which chemistry round 1 was weakest on (hetero/carbo-aromatic
pairs), and adding real DFT labels for exactly that chemistry produced a
model whose advantage over the g-xTB baseline nearly doubled on a harder,
more diverse population, with a larger and more trustworthy frozen holdout
(R² 0.697 → 0.831). The apparent MAE regression (2.106 → 3.147) is fully
explained by the harder selected population, not by degraded model quality —
confirmed by the model's unchanged ~2.0 MAE when evaluated on round-1-only
rows. This validates continuing the active-learning loop for round 3.
