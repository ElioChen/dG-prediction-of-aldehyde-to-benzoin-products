# Better-label probe, HARD-TAIL TARGETED (K=10 conformers) — 20260902_1738

Corrected version of the 2026-06-26 pilot (see descriptor-search-exhausted memory's 2026-07-10 correction): targets the sulfonyl/P/imine/amide top-|error| subset of the CURRENT champion's test set (not a random draw of easy molecules), K=10 conformers (not 5), scored against the current champion's own dG_pred (not a stale June model).


n molecules (both species ok): **150** of 150 targeted hard-tail sample; wB97X-3c ok: 20

## How much do labels move? (kcal/mol)
| shift | mean | std | mean|·| | 90th pct |·| | max |·| |
|---|---|---|---|---|---|
| Boltzmann correction (boltz−single) | -0.131 | 2.548 | 1.732 | 4.323 | 9.190 |
| functional shift (wB97X−r2SCAN) | -0.074 | 3.558 | 2.843 | 5.371 | 8.270 |

## Does re-labeling lower the CURRENT champion's MAE on this hard-tail sample? (frozen predictions)
| label set | model MAE |
|---|---|
| stored single-conformer r2SCAN-3c (current) | **10.011** |
| Boltzmann-averaged r2SCAN-3c (K=10) | **10.126** (+0.115) |
| wB97X-3c single-conformer | **9.434** (-0.577) |

## Interpretation
Multi-conformer Boltzmann re-labeling **does NOT lower** the champion's measured MAE on this hard-tail sample (10.01→10.13). Caveat unchanged from the original pilot: this re-scores FROZEN predictions rather than retraining, so it only tests whether label noise is inflating apparent error on molecules the model already got wrong -- it does NOT directly measure whether a model retrained on relabeled data would generalize better. If this LOWERS MAE meaningfully (materially more than the -0.05 threshold), that is real evidence the label-noise lever is worth pursuing at scale (full-tail relabel + retrain); if it does not, the two independent nulls (random-draw K=5 and targeted-hard-tail K=10) would jointly rule out this lever and it should be dropped for good.
