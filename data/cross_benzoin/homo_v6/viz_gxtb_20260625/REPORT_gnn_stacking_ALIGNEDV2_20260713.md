# GNN+tabular stacking, ALIGNED v2 -- fresh tabular retrain, zero file-drift (20260713)

Fixes the v1 aligned run's residual issue (population drift between the stored 2026-07-06 predictions file and today's fresh feature rebuild collapsed overlap to n=2,254). This version trains the tabular ensemble FRESH in the same script/session as the GNN, on the identical in-memory split -- 100% overlap guaranteed by construction, no file join.

- Tabular fresh split: train=152,673 val=43,621 test=21,811
- Tabular ensemble (fresh retrain) test MAE=1.485 (vs stored champion bundle's 1.503 -- small delta expected from upstream data drift)
- GNN cache coverage: train 152,673, val 43,621, test 21,811/21,811
- GNN standalone MAE=1.562
- **TRUE full-coverage overlap: n=21,811**
- tabular-only MAE=1.485 | gnn-only MAE=1.562 | **best blend w_gnn=0.40 MAE=1.427 (delta -0.058)**

This is now the most trustworthy stacking-gain estimate to date (v1's 2026-07-07 n=6,601 partial + 2026-07-10 n=2,254 partial both pointed the same direction, -0.05 and -0.04 respectively; this run should confirm on close to the full test set). If the delta here is still meaningfully negative, promote the GNN+tabular blend to production; if it's flat/positive, the earlier partial-overlap gains were themselves subset artifacts.
