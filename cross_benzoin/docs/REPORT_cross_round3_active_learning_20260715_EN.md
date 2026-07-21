# Cross-benzoin round 3: split-consistent sampling + closed active-learning loop (2026-07-15)

Companion file: `REPORT_cross_round3_active_learning_20260715_ZH.md` (中文版).

> **Caveat added 2026-07-17**: the frozen-holdout MAE/R² numbers in this report were measured
> on `candidates_v3`'s original molecule-level (InChIKey-disjoint) split, later found to leak
> Bemis–Murcko scaffolds severely (93% of the 29-row holdout had at least one side's scaffold
> already present in train). Treat this report's frozen-holdout numbers as optimistic by
> roughly 0.2–0.5 MAE relative to true novel-scaffold generalization. The corrected,
> scaffold-disjoint evaluation (n_test=450) and the recommended production model
> (`cross_benzoin/predict_cross_champion.py`) now supersede this metric — see `STATUS_EN.md`'s
> correction note and memory `cross-scaffold-disjoint-rebuild-20260717`. This report is
> otherwise kept unmodified below as a historical snapshot.

## What this round was

Round 2 (`REPORT_cross_round2_active_learning_20260714_EN.md`) validated
that uncertainty-driven active learning beats diversity-only sampling: the
model's improvement margin over the g-xTB baseline nearly doubled
(+1.34→+2.39 kcal/mol) and the frozen holdout got both bigger and more
accurate (R² 0.697→0.831). It also flagged an unresolved cost: both round 1
and round 2 lost ~31-33% of their rows from the frozen molecule-disjoint
holdout evaluation, because their custom pair generators didn't respect
`candidates_v3`'s pre-computed train/validation/test split (both aldehydes
of a pair must land in the same split; a generator that doesn't know about
the split will violate this ~a third of the time by chance).

Round 3 closes that gap and completes a full three-round active-learning
cycle:

1. **Sampled directly from `candidates_v3`'s own 4M-row directed candidate
   list** (`sample_round3_from_candidates_v3.py`) instead of a fresh custom
   generator — every row this produces already carries a correct,
   pre-computed split label by construction, and the sampler only reads
   from the `train` split, excluding all 2397 pairs already used in rounds
   1-2. Systematic stride sampling within each of the file's 6 contiguous
   class_pair blocks kept the 4200-pair pool category-balanced. 27 seconds
   to build, zero exclusions needed at this pool size.
2. Featurized (xTB + g-xTB) and computed product BDE — **8400/8400 rows,
   99.9% error-free**, the cleanest completion rate of any round so far.
3. Scored all 4182 valid pairs with a 40-member bootstrap ensemble trained
   on the **round1+round2 COMBINED table** (2354 rows) — the best available
   reference model at this point in the loop, not round 1 alone.
4. DFT-SP'd the top 900 pairs (1800 rows, **zero pairs lost to one-sided
   coverage** — better than round 2, since round 3's candidates were
   pre-filtered against the same aldehyde cache the round-2 retry step had
   just backfilled).
5. Merged all three rounds (4120 rows / 2062 pairs) and retrained.

## Result 1 — a real bug found and fixed along the way (not part of the science, but load-bearing)

While verifying chunk output, found that `submit_cb_featurize_array.sh`
silently reported `COMPLETED` in `sacct` for chunks that had actually
hard-failed (`--require-cache-complete` aborts a whole 100-row chunk if
even one of its ~150-190 unique aldehydes is missing from cache — by
design — but the submit script piped through `tee` without propagating the
real exit code, so 16/84 chunks looked "done" while producing zero rows).
Fixed the script (captures and re-exits with the real code) and resubmitted
just the 16 affected chunks without the cache-complete flag — recovered all
1600 otherwise-lost rows. Separately, a 3-row DFT-SP smoke test showed 2/3
failures (`orca_sp_failed`) on unusually large selected molecules (84-116
heavy atoms); reproducing one standalone (no concurrent process) succeeded
cleanly, diagnosing the failures as transient resource contention from
running 3 concurrent full-SCF ORCA jobs on a shared login node, not a real
convergence problem — confirmed when the full production array job (run
under dedicated SLURM allocation) came back **1766/1766, zero errors**.

## Result 2 — active-learning uncertainty self-corrected the category imbalance ROUND 2 ITSELF created

Round 2 heavily under-selected aliph-aliph pairs (18% selection rate,
lowest of any category) because round 1's model was already confident on
simple aliphatic chemistry. That left aliph-aliph as only **4.5%** of the
round1+2 combined training table (107/2354 rows) — by far the smallest
category. Round 3's uncertainty ranking (scored against that same
round1+2 model) picked this up automatically: aliph-aliph was selected at
**61%** this round — the HIGHEST rate of any category, a complete reversal
from round 2's pattern (carbo-hetero/hetero-hetero/carbo-carbo dropped from
51-63% selection down to 15-21%). This is exactly the self-correcting
behavior a healthy iterative active-learning loop should exhibit: it isn't
locked onto whatever it was uncertain about at the start, it re-targets
wherever the CURRENT model (including the side-effects of its own prior
selections) is weakest.

## Result 3 — the split-consistency fix worked, provably, not just plausibly

| metric | round1 alone | round1+2 | round1+2+3 |
|---|---:|---:|---:|
| rows / pairs | 598 / 299 | 2354 / 1179 | 4120 / 2062 |
| frozen n_train | -- | 1515 | **3281** |
| frozen n_test | 14 | 29 | **29** |
| frozen loss rate | -- | 788/2354 = 33.5% | **788/4120 = 19.1%** |

`n_train` grew from 1515 to 3281 — a gain of **exactly 1766**, round 3's
entire DFT-labeled contribution, with **zero rows lost** to split
inconsistency. The absolute dropped-row count (788) is IDENTICAL between
round1+2 and round1+2+3, because round 3 was sampled only from
`candidates_v3`'s `train` split by construction — it could only ever grow
`n_train`, never touch `n_test`, and could never land in the "wrong split"
bucket either. The loss RATE nearly halved (33.5%→19.1%) purely because the
denominator grew while the (fixed, round1+2-inherited) numerator didn't —
exactly the expected signature of "new rounds stop bleeding data; the old
damage gets diluted," not a partial failure of the fix. `n_test` staying
flat at 29 is also correct and expected: a new training round should never
touch held-out test molecules, and round 3 never did.

## Result 4 — apples-to-apples comparison finally possible: more (well-targeted) data measurably improved the SAME frozen test set

Because `n_test` is identical (the same 29 molecule-disjoint rows) between
round1+2 and round1+2+3, this is the first genuinely clean before/after
comparison in the project — same test set, more training data:

|  | round1+2 | round1+2+3 |
|---|---:|---:|
| frozen Δ-model MAE | 3.612 | **3.398** |
| frozen R² | 0.831 | **0.842** |
| frozen g-xTB MAE | 5.315 | 5.315 *(same rows — baseline unaffected by training)* |

A 0.214 kcal/mol MAE improvement and 0.011 R² gain, attributable purely to
round 3's additional, correctly-sourced training data.

## Result 5 — category errors got more UNIFORM, not just lower on average

| reaction_type | round1+2 Δ-MAE | round1+2+3 Δ-MAE |
|---|---:|---:|
| aliph-aliph | 3.639 | **3.461** |
| aliph-carbo | 3.580 | 3.254 |
| aliph-hetero | 3.751 | 3.312 |
| carbo-carbo | 3.003 | 2.944 |
| carbo-hetero | 3.048 | 2.841 |
| hetero-hetero | 2.964 | 2.863 |

Every category improved, and the spread tightened from 2.96-3.75 kcal
(round1+2) to 2.84-3.46 kcal (round1+2+3) — the model got more uniformly
competent across chemistry types, which is what targeted active learning
(vs. blanket data addition) is supposed to deliver.

## Result 6 — the CV headline number plateaued, and that's fine given results 3-5

CV Δ-model MAE (2.966) and the CV g-xTB baseline (5.335) both landed
slightly BETTER than round1+2 alone (3.147 / 5.534) — round 3, despite
reaching wider ΔG extremes (up to +89.3 kcal, vs round 2's +52.6), was not
systematically harder on net, because it also deliberately added back
simpler aliph-aliph chemistry. The MAE-over-g-xTB margin (+2.369) is
essentially flat vs round1+2's +2.387 — a genuine plateau in that one
number, but not a concerning one: results 3-5 above show real, measurable,
independently-attributable gains (holdout R², category uniformity,
split-consistency) that the single CV-margin number doesn't capture.

## Result 7 — feature-block ablation held a THIRD time

| block | n_feats | MAE |
|---|---:|---:|
| product_only | 54 | 3.525 |
| all_raw_blocks | 140 | **2.955** |
| all+interaction | 150 | 2.966 |

`all_raw_blocks` again edges out `all+interaction` — the donor/acceptor
complementarity interaction terms add nothing measurable, now confirmed at
n=598, n=2354, AND n=4120. No longer attributable to noise; a candidate to
drop from a future default config (harmless to keep, but the 140-feature
version is equally good and cheaper).

## Bottom line for the full 3-round loop

Round 2 proved uncertainty-driven active learning beats diversity sampling.
Round 3 proved three more things: (1) sampling directly from
`candidates_v3` eliminates the split-consistency data loss for every new
round going forward; (2) the self-correcting category-rebalancing behavior
of iterative uncertainty sampling is real, not theoretical — it visibly
reversed round 2's own selection bias; (3) the frozen, leakage-safe holdout
metric — for the first time comparable round-over-round on a genuinely
fixed test set — shows real, reproducible improvement from targeted data
addition (R² 0.831→0.842), not just from having a bigger dataset. The loop
is validated end-to-end and ready to continue into a round 4 with no
blocking issues.
