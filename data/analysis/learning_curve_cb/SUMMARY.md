# Learning curve — cb Δ-model label-set sizing (2026-06-24)

Data: `featurize_cb_homo_train_gxtb.parquet` (funnel_v3 r2SCAN-3c labels, g-xTB baseline,
aldehyde+product QM descriptors, **all CHO categories**, 2,900 rows). Fixed 10% held-out
test; train on increasing n; fixed xgb hyperparams (n_estimators=800, depth=5, lr=0.03);
5 repeats/point. Script `pipeline/learning_curve_cb.py`.

## Result
test MAE: 200→3.04, 600→2.89, 1200→2.75, **1600→2.69 (min)**, 2000→2.70, 2500→2.77,
2610→2.72. g-xTB baseline (no model) = 3.81.

## Conclusions
1. **Plateau ≈ 1,500–1,600 labels** — flat/noisy beyond. The 3,050 labels are overkill for
   this model; ~1,500 well-chosen molecules suffice. (Silhouette k-means could NOT size this;
   the chemical space is continuous, silhouette ~0.05 at all k — see `data/analysis/subset_v6/`.)
2. **Δ-model beats raw g-xTB by only ~1.1 kcal** (3.81→2.7) — g-xTB already strong for benzoin.
3. **All-categories + untuned + no-QC + holdout gives ~2.7**, vs the earlier aromatic-only +
   tuned + corr-MAD QC + CV ~2.0. Including aliphatic + dropping QC costs ~0.7 kcal (expected).

## Implications / next options
- For the ML route, no need for more DFT labels; ~1,500 is enough. A principled re-derived
  subset would use diversity/active-learning to pick ~1,500 on the v6 library, not silhouette-k.
- The full-library funnel_v3 DFT-SP (running, array 24176554) makes a subset moot for the 220k
  itself (we get exact DFT for all), but matters for cheap future candidates outside the library.
- To recover accuracy under the all-category scope: re-run with hyperparameter tuning + QC.
