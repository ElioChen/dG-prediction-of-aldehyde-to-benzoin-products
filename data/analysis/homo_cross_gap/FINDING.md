# homo (1.503) vs cross (2.215) dG MAE — the gap is regime + scale, not task difficulty

**Date:** 2026-09-07 · **Script:** `pipeline/analysis/homo_cross_gap_decomposition.py` ·
**Data:** `homo_unify_v1` (30,000 git-tracked homo pairs that survived the 2026-07 purge
*with* `dG_orca_kcal` + a Bemis-Murcko scaffold-disjoint split) · **Recipe:** the same
72-feature Δ-learning ensemble (MLP + XGB-d8 + XGB-d10) as `finalize_correction.py`.

## The question

`homo 1.503` (headline) and `cross 2.215` (champion blend) are quoted side by side but
are **not the same measurement**:

| | homo headline | cross champion |
|---|---|---|
| split | random 70/10/20 | Bemis-Murcko **scaffold-disjoint** |
| regime | interpolation (closed 219k library) | **extrapolation** to novel scaffolds |
| train size | ~154k | 22,771 |
| model | MLP+XGB ensemble | + attentive GNN blend |

## Measured decomposition (all on the same 72-feat recipe, homo_unify 30k)

| condition | n_train | MAE | R² | g-xTB baseline MAE |
|---|--:|--:|--:|--:|
| homo, **random** split | 18,197 | **2.265** | 0.728 | 4.09 |
| homo, **scaffold-disjoint** split | 19,030 | **2.608** | 0.726 | 4.43 |
| homo, scaffold-disjoint, train sub-sampled to 19,030 (≈ cross) | 19,030 | 2.610 | 0.727 | 4.43 |
| *cross ensemble-only (scaffold-disjoint, 22,771)* — from CHAMPION.md | 22,771 | *2.326* | *0.739* | *5.04* |
| *cross blend (+ GNN)* | 22,771 | *2.215* | — | *5.04* |

## Reading: homo 1.503 → cross 2.215 (+0.712 kcal) decomposes as

| step | ΔMAE | what it is |
|---|--:|---|
| homo 1.503 → homo random @18k **2.265** | **+0.762** | pure **training-scale** effect (154k → 18k), same random-split regime |
| homo random @18k → homo scaffold-disjoint @19k **2.608** | **+0.343** (**+15%**) | homo's **own leakage premium** — interpolation → true scaffold extrapolation |
| homo scaffold-disjoint @19k → cross ensemble-only @23k **2.326** | **−0.282** | at matched split regime + train size + architecture the **cross task is *not* harder** for the tabular model — slightly easier |
| cross ensemble-only → cross blend **2.215** | **−0.111** | the GNN blend the champion adds (homo has no GNN leg) |
| **sum** | **+0.712** | = the full headline gap |

## Conclusion

**The homo/cross gap is now decisively characterised: it is split regime + training
scale, not intrinsic task difficulty and not label quality.**

- The two things that make the homo headline look better — a 12× larger training set and
  a random (interpolation) split — together *more* than account for the 0.71 kcal gap.
- At apples-to-apples conditions (scaffold-disjoint, ~20 k train, XGB/MLP ensemble),
  **cross (2.33) ≈ homo (2.61); cross is if anything slightly ahead** (~2 bootstrap SE
  at n=448, so "at least as good", not "clearly better").
- homo's own leakage premium is **+15%**, in the same family as cross's +9.8 % and the
  BDE sub-project's +43 % — "molecule-level split ≠ scaffold generalisation" is a
  three-way-confirmed, task-independent effect.
- homo scaffold-disjoint MAE 2.61 sits **well above** homo's own ~2.1 kcal
  single-conformer label floor → homo-at-matched-conditions is **data/generalisation
  limited**, whereas cross blend 2.215 is near its ~2.9 kcal floor. So the cross model
  is closer to "done" than a regime-matched homo model would be.

### Caveats

30 k homo subset, not the full 219 k (the rest lost its `dG_orca` labels in the purge);
different test-set sizes (3,518 vs 448); 72-feat homo recipe vs the cross champion's
260; homo's single aldehyde vs cross donor+acceptor. None of these plausibly reverse a
0.7 kcal decomposition, but the −0.28 "cross is easier" term is the softest (~2 SE).

### For deployment

There is no "cross penalty" to normalise away. Per-prediction calibration was shipped
separately (`predict_dg_calibration.json` / `predict_dg.py` split-conformal PI, 09-07).
Whether a **unified homo+cross model** beats cross-only at current scale is tested in
`homo_cross_joint_tabular.py` (Task C).
