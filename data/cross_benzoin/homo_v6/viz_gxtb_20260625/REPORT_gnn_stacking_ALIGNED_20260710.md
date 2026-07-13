# GNN+tabular stacking, PROPERLY ALIGNED split (20260710)

Tier-1c of the 2026-07-10 external-diagnosis review. Fixes the population-mismatch in the 2026-07-07 attempts (jobs 24482531/24489591, see gnn-delta-result memory) by deriving the GNN's train/val/test ids directly from the tabular champion's exact population+split, then training the GNN from scratch on those ids (reusing the existing cached graphs, no rebuild).

- Tabular authoritative split: train=152,673 val=43,621 test=21,811
- Cache coverage: train 152,673 (100.0%), val 43,621, test 21,811 (100.0%)
- GNN standalone (aligned test set): MAE=1.589 RMSE=2.336 R2=0.867
- **Final overlap for blend comparison: n=2,254** (vs the prior partial attempt's n=6,601 -- 0.3x larger, much closer to the tabular model's full test n=21,910)
- tabular-only MAE=1.470 | gnn-only MAE=1.623 | **best blend w_gnn=0.30 MAE=1.431 (delta -0.039)**

If this delta is still negative (blend beats tabular-only) and now validated on ~full test-set coverage, the stacking gain from 2026-07-07 is CONFIRMED and worth promoting to production (average GNN+tabular predictions at the best w). If the delta shrinks toward zero or flips sign now that the comparison isn't restricted to an easier ~30% subset, the earlier -0.051 was itself a subset-selection artifact, not a real generalizable gain.
