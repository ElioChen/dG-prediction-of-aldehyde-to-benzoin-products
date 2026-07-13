# baseline_72 overfitting / robustness / reproducibility (20260702)

## Overfitting (train vs val vs test, per reshuffled seed)

| seed | train MAE | val MAE | test MAE | test-train gap |
|---|---|---|---|---|
| 0 | 1.011 | 1.585 | 1.559 | 0.548 |
| 1 | 1.013 | 1.567 | 1.569 | 0.555 |
| 2 | 1.045 | 1.577 | 1.587 | 0.543 |
| 3 | 1.054 | 1.585 | 1.584 | 0.530 |
| 4 | 0.952 | 1.553 | 1.555 | 0.603 |

mean gap = 0.556

## Seed sensitivity (5x reshuffled 70:20:10, same ratios)

test MAE = 1.571 +/- 0.013 (range 1.555-1.587)

## 5-fold cross-validation

| fold | train MAE | val MAE | test MAE |
|---|---|---|---|
| 0 | 1.033 | 1.567 | 1.586 |
| 1 | 1.009 | 1.570 | 1.561 |
| 2 | 1.057 | 1.610 | 1.583 |
| 3 | 1.022 | 1.572 | 1.572 |
| 4 | 1.054 | 1.588 | 1.580 |

CV test MAE = 1.577 +/- 0.009 vs holdout mean 1.571
