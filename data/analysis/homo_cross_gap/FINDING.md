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

![homo/cross gap waterfall](../../../docs/figures/homo_cross_gap_waterfall.png)

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

---

## Task C — does a unified homo+cross model help at current scale?

`cross_benzoin/homo_cross_joint_tabular.py` + an ensemble-level check
(`homo_cross_joint_ensemble_check.json`). Δ = `dG_orca − dG_gxtb`, 72-feat shared space,
homo_unify 26k + cross clean-train 23k, eval on the cross scaffold-disjoint holdout (n=448).

| condition | single-XGB MAE | MLP+XGB ensemble MAE |
|---|--:|--:|
| `cross_only` | 2.812 | 2.716 |
| `naive_merge` (+homo, +is_homo flag) | **2.688 (−0.124)** | **2.610 (−0.106)** |
| `finetune` (homo → cross continue-train) | 2.849 (null) | — |
| `homo_only_zeroshot` | 3.109 | — |

![homo+cross joint tabular](../../../docs/figures/homo_cross_joint.png)

**Read: a modest but reproducible gain.** `naive_merge` improves ~0.11 kcal on *both* a
weak single-XGB and a proper ensemble — so it is not just "a data-starved model likes
more rows". `finetune` (the "proper" transfer) is null. This is the opposite of the
pre-purge BDE-side finding (`finetune > cross-only > naive_merge`), because here the
homo:cross ratio is ~1.1:1 (balanced) so pooling doesn't dilute the cross signal — at
full scale (219k homo : 35k cross ≈ 6:1) the dilution would return unless homo is
down-weighted.

**Caveats:** 72-feat proxy is ~0.5 kcal weaker than the 260-feat + GNN champion (2.716
vs 2.215), so the gain may shrink against the champion; n_test=448 → each −0.11 is <1
bootstrap SE (the strength is the *direction* holding across two model classes, not any
single number).

**Verdict: AMBER-GREEN — worth a bounded next step, not a blank cheque.** Build the full
260-schema featurization for the 30k homo_unify products (mordred + assemble, ~1-2 day
pipeline), retrain the cross ensemble + GNN with homo included (down-weighted to ~1:1),
and check whether the −0.1 survives to the champion. If it does → then invest in the
GNN homo-pretrain→finetune path (which the current champion does NOT use — it was never
rebuilt after the 2026-07 purge).

---

## Task D — would a better cheap Δ-learning baseline lower the floor?

`cross_benzoin/cheap_baseline_pilot_worker.py` + `merge_cheap_baseline_pilot.py`. The
DFT-arbitration work showed the g-xTB↔r2SCAN-3c gap is dominated by the single-point
method level (|Δ_SP| ~16 vs |Δ_geom| ~5), so the one untested accuracy lever is a
better-but-still-cheap single point as the Δ-learning baseline: g-xTB (semiempirical) →
**B97-3c** (GGA composite, ~5-20× cheaper than the r2SCAN-3c label).

128 pairs (64 heteroatom hard-tail + 64 control from the holdout), one fresh GFN2
geometry per species, then g-xTB / B97-3c / r2SCAN-3c single points on that identical
geometry (so `resid_gxtb` vs `resid_b973c` is conformer-noise-free). 128/128, 0 errored.

![cheap-baseline pilot](../../../docs/figures/cheap_baseline_pilot.png)

| residual `dG_r2scan − dG_baseline` | mean\|·\| | **std** | mean (signed) | p90\|·\| |
|---|--:|--:|--:|--:|
| g-xTB baseline | 6.00 | **4.32** | +5.75 | 11.2 |
| B97-3c baseline | 5.24 | **1.11** | **−5.24** | 6.7 |

The B97-3c residual is a near-constant −5.24 kcal offset (mean|·| ≈ |mean_signed| → the
scatter around the offset is tiny) + std ≈ 1.11. A Δ-learning model trivially absorbs a
constant offset, so **the achievable floor is set by the std**: g-xTB 4.32 → B97-3c
**1.11** (std ratio **0.26**). On the hardest heteroatom hard-tail (g-xTB std 5.1)
B97-3c still holds std 1.2.

`merge_cheap_baseline_pilot.py` mechanically returns **AMBER** (its GREEN gate also
requires mean|·| to drop, but mean|·| is dominated by the −5.24 constant offset, which
is absorbable) — **substantively closer to GREEN**: the std ratio 0.26 is decisive.
Switching the baseline to B97-3c drops the Δ-model's theoretical floor from ~4.3 to
~1.1, potentially pushing the champion MAE well below 2.215 — the first real accuracy
lever in months.

**Caveats:** the one-shot ETKDG/GFN2 geometry is ~18 kcal off the production funnel_v3
labels; and with all three SPs on one geometry, the 1.11 std is pure level-of-theory
scatter with no conformer noise (production adds that on top). The B97-3c↔r2SCAN-3c
constant-offset relationship is a level-of-theory property and likely geometry-robust,
but this must be confirmed.

**Next steps:** (1) re-run ~30 pairs on production funnel_v3 geometries to confirm the
low-scatter property; (2) if confirmed → a full B97-3c-baseline recompute (35k pairs ×
3 species) + retrain the Δ-model on the B97-3c baseline, and measure whether the
champion MAE drops materially. Raw per-pair data:
`data/cross_benzoin/cheap_baseline_pilot/cheap_baseline_pilot_merged.csv`.
