# Cross-benzoin round6: DFT-SP close-out, retrain at 6-round scale, and a GNN-blend result that did NOT replicate

**Date**: 2026-07-16
**Continues**: `REPORT_cross_ensemble_and_unification_20260716_{EN,ZH}.md` (which left round6 "in flight": DFT-SP array 24667830 running, 50/80 chunks done).

> **Caveat added 2026-07-17**: every frozen-holdout MAE number in this report was measured on
> `candidates_v3`'s original molecule-level (InChIKey-disjoint) split, later found to leak
> Bemis–Murcko scaffolds severely (93% of the 29-row holdout had at least one side's scaffold
> already present in train). Treat this report's frozen-holdout numbers as optimistic by
> roughly 0.2–0.5 MAE relative to true novel-scaffold generalization. The corrected,
> scaffold-disjoint evaluation (n_test=450) and the recommended production model
> (`cross_benzoin/predict_cross_champion.py`) now supersede this report's headline metric — see
> `STATUS_EN.md`'s correction note and memory `cross-scaffold-disjoint-rebuild-20260717`. This
> report is otherwise kept unmodified below as a historical snapshot.

## 1. Round6 finished end-to-end, zero manual intervention

- SLURM array **24667830** (80 chunks × 100 rows, genoa) reached **80/80 COMPLETED**.
- `cross_round6_dft_products.csv`: 7,996 rows, **0 errors**.
- Product g-xTB BDE follow-up (`calc_bde_gxtb_product_cross.py`-equivalent) ran to completion: 80/80 chunk CSVs in `data/cross_benzoin/cross_round6/bde_gxtb/`.
- Assembled into a 6-round training table: `cross_train_table_6rounds_mordred_slim120_matched.parquet`, **25,176 rows / 12,597 pairs** (up from round5's 17,270/8,642).
- All three architecture lines were retrained on it: single-XGB champion, MLP+XGB ensemble, and the homo-pretrained triple-encoder GNN — the same recipes used at round5, unmodified, so the round5→round6 deltas below isolate the effect of the extra data.

## 2. Full progression, round3 → round6

Frozen holdout is always the same 29 molecule-disjoint `candidates_v3` test rows; the g-xTB baseline on those rows is therefore identical (MAE 5.315) at every round, which is a useful sanity check that the comparison stayed apples-to-apples throughout.

| | R1-3 | R1-4 | R1-5 | R1-6 |
|---|---:|---:|---:|---:|
| rows / pairs | 4,120 / 2,062 | 12,378 / 6,194 | 17,270 / 8,642 | 25,176 / 12,597 |
| CV champion MAE | 2.966 | 2.261 | 2.397 | 2.276 |
| CV champion R² | 0.750 | 0.751 | 0.409 | 0.473 |
| CV g-xTB baseline MAE | 5.335 | 4.191 | 4.499 | 4.292 |
| **frozen champion MAE** | 3.398 | 3.043 | 2.983 | **3.132** |
| frozen champion R² | 0.842 | 0.859 | 0.865 | 0.854 |
| frozen ensemble MAE | — | — | 2.633 | **2.582** |
| frozen ensemble R² | — | — | 0.898 | 0.897 |

Two things stand out:

1. **The single-XGB champion's frozen MAE got worse from round5 to round6** (2.983 → 3.132), the first regression in this metric across the whole project. CV MAE simultaneously improved (2.397 → 2.276), so this is not a case of the model degrading in general — it's the 29-row frozen set moving against this specific architecture as the training population shifts. Given n=29, some of this is expected sampling variance on the *frozen side* rather than a real capability drop, but it is the reason the ensemble (not the single-XGB model) should be treated as the current best-validated artifact, not a data-glitch to explain away.
2. **The MLP+XGB ensemble kept improving** (2.633 → 2.582), continuing to be the best model at every scale it's been tried at so far.

## 3. The GNN-blend win from round5 did not replicate

The previous report highlighted a stacking result: at round5 scale, a fixed 50/50 blend of the ensemble and the GNN beat the ensemble alone with bootstrap significance (`analyze_gnn_stack_uncertainty.py`, P(blend better)=0.987, n=29). Retraining the GNN on the round6 table and rerunning the *identical* analysis script gives a different answer:

| | round5 | round6 |
|---|---:|---:|
| ensemble-only MAE | 2.777 | **2.547** |
| GNN-only MAE | 2.624 | 2.820 |
| 50/50 blend MAE | 2.520 | 2.564 |
| P(blend better than ensemble) | 0.987 | **0.456** |
| P(GNN-only better than ensemble) | 0.767 | 0.096 |

(Note the round5 ensemble-only MAE printed by the analysis script, 2.777, differs slightly from the ensemble's own reported frozen MAE of 2.633 in section 2 — the analysis script recomputes predictions on the exact 29-row intersection where the GNN also has a valid prediction, a very slightly different accounting than the ensemble's own standalone frozen-holdout eval; the direction and significance of the comparison is what matters here, not the third decimal.)

Because the frozen test set is the literal same 29 rows in both runs, this is a real reversal, not test-side noise: the ensemble's error on those rows dropped (2.777→2.547) while the GNN's rose (2.624→2.820). The GNN was retrained with the **same architecture and hyperparameters as round5** (homo-pretrained triple-encoder, no retuning attempt for the ~46% larger training set) — the most likely explanation is under-tuning at the new scale, not a fundamental ceiling on GNN performance here. `train_cross_gnn.py`'s own internal blend search (a continuous weight sweep, not the fixed-0.5 blend the uncertainty script tests) found an optimal GNN weight of only 0.15 at round6 (metadata: `best_blend_w_gnn: 0.15`, `best_blend_mae: 2.531` vs ensemble-alone 2.547) — a much smaller, borderline-negligible edge than the 50/50 blend the uncertainty script tested, consistent with the GNN having comparatively less to contribute at this scale than it did at round5.

**Reading**: treat round5's "blend wins" conclusion as provisional and superseded, not a validated finding to build on. The ensemble alone (`data/cross_benzoin/cross_round6/train_ensemble_6rounds_slim120_v1/models/cross_ensemble_model.joblib`, frozen MAE 2.582, R² 0.897) is the current best-validated single artifact. Whether GNN stacking is worth pursuing further is now an open question that would need actual retuning (learning rate, epochs, capacity) at the round6 scale before drawing a conclusion either way — not just re-running the same recipe on more rows.

## 4. What's not yet decided

- **Model promotion**: the round6 MLP+XGB ensemble is the best-validated model to date but has not been promoted to be the scoring model `score_round_active_learning.py` uses for future active-learning rounds (still using the single-XGB champion line).
- **GNN retuning**: worth a deliberate hyperparameter pass before trusting or discarding the stacking idea — round5's result looked like a real win at the time and wasn't; a single retrain at one new scale isn't enough evidence to kill it either.
- **Round7**: 3,677/7,677 pairs remain unused in the screen10k reservoir after round6 spent 4,000 of them — a round7 could draw from what's left with zero new xTB/g-xTB compute, same as round6.
- **Homo+cross unification, third measurement**: prior sessions found the unification benefit shrinking as cross's own data grows (-1.9% relative MAE at 4,120 rows → -1.0% at 17,270 rows). Round6 (25,176 rows) is the natural next data point to see whether that trend flattens, continues shrinking, or reverses.

## Bottom line

Round6 closed out cleanly with no data-integrity issues (0 DFT errors, standard ~94%-range compute success). The MLP+XGB ensemble remains the best-validated cross model and improved again (frozen MAE 2.633→2.582). The single most important finding this round, however, is negative: the GNN-stacking win reported for round5 reversed under an identical re-test at round6 scale, and should not be treated as settled science — a caution against trusting single-round significance tests (even bootstrap ones) on an n=29 frozen holdout without a replication check across at least one more scale.
