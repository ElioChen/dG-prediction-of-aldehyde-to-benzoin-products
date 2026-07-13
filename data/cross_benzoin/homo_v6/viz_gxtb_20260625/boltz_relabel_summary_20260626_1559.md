# Better-label probe — Boltzmann DFT & wB97X-3c — 20260626_1559

n molecules (both species ok): **95** of 120 test-split; wB97X-3c ok: 71

## How much do labels move? (kcal/mol)
| shift | mean | std | mean|·| | 90th pct |·| | max |·| |
|---|---|---|---|---|---|
| Boltzmann correction (boltz−single) | 0.113 | 0.916 | 0.616 | 1.287 | 3.520 |
| functional shift (wB97X−r2SCAN) | -0.490 | 1.685 | 1.413 | 2.621 | 5.726 |

## Does re-labeling lower the model MAE on these 120? (same frozen predictions)
| label set | model MAE |
|---|---|
| stored single-conformer r2SCAN-3c (current) | **0.970** |
| Boltzmann-averaged r2SCAN-3c | **1.284** (+0.314) |
| wB97X-3c single-conformer | **1.500** (+0.530) |

## Interpretation
Multi-conformer Boltzmann re-labeling **does NOT lower** the measured MAE (0.97→1.28). The Boltzmann correction has std 0.92 and mean magnitude 0.62 kcal/mol — that is the label movement available from conformer averaging. The functional swap (wB97X-3c vs r2SCAN-3c) moves labels by std 1.69 (mean |·| 1.41) — a systematic+random bias of the chosen functional. If either MAE drops materially below 1.6, single-conformer/functional label noise is a real component of the floor and a full multi-conformer (or higher-functional) re-label is justified; if not, the floor is intrinsic model/feature error. See [[delta-mae-noise-floor]], [[conformer-search-noise]], [[dft-labels-r2scan-not-pbe0]].
