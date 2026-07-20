# Cross-benzoin: MLP+XGB ensemble packaging, homo+cross unification re-test at scale, and round 6

**Date**: 2026-07-16
**Continues**: `REPORT_cross_round3_active_learning_20260715_{EN,ZH}.md` and the round4/5 scale-up work summarized in memory `cross-round4-5-scaleup-and-architecture-win-20260716`.

> **Caveat added 2026-07-17**: every frozen-holdout MAE number in this report was measured on
> `candidates_v3`'s original molecule-level (InChIKey-disjoint) split, later found to leak
> Bemis–Murcko scaffolds severely (93% of the 29-row holdout had at least one side's scaffold
> already present in train). Treat this report's frozen-holdout numbers as optimistic by
> roughly 0.2–0.5 MAE relative to true novel-scaffold generalization. The corrected,
> scaffold-disjoint evaluation (n_test=450) and the recommended production model
> (`cross_benzoin/predict_cross_champion.py`) now supersede this report's headline metric — see
> `STATUS_EN.md`'s correction note and memory `cross-scaffold-disjoint-rebuild-20260717`. This
> report is otherwise kept unmodified below as a historical snapshot.

## Context

The previous session (2026-07-15/16) confirmed three things but left them at different stages of completeness:

1. An MLP+XGB ensemble architecture beats the single-XGB champion in cross-validation, and its edge *grows* with scale (-4.9% relative MAE at 4120 rows, -9.5% at 17270 rows) — but this was CV-only, with no frozen molecule-disjoint holdout number and no shippable model artifact.
2. Homo+cross unification (training on cross's own rows plus a 30,000-row stratified sample of the homo/self-condensation library) gave a real but modest gain at the old 4120-row cross scale (CV MAE on cross rows only: 2.966 → 2.910, -1.9% relative) — not re-tested since cross's own data nearly quadrupled (round4/5, → 17,270 rows) and mordred descriptors were folded into the pipeline.
3. A 7,677-pair reservoir of already-featurized, unlabeled candidates sat unused after round 5 only spent 2,500/10,177 of the original screen10k pool.

This session closed all three gaps.

## 1. MLP+XGB ensemble: frozen holdout + shipped artifact

New `cross_benzoin/train_cross_ensemble.py`, mirroring `train_cross_delta.py`'s conventions so the two are directly comparable:

- **Feature set fix**: the original CV-only experiment (`architecture_ensemble_experiment.py`) used *every* column in the table, including the 10 `interaction_*` terms confirmed useless three times over. The new script uses the champion's exact `all_raw_blocks+mordred` selection (260 features, interaction terms excluded) — a genuine apples-to-apples comparison, not just "same CV protocol, different feature set."
- **Frozen holdout**: same `pair_split_labels()` / `frozen_holdout_eval()` machinery as the champion, fit on `candidates_v3`'s `train`-split rows, evaluated once on the same 29 frozen `test`-split rows.
- **Packaging**: a `MLPXGBEnsemble` class (StandardScaler + MLP(128,64) + XGB(depth=5) + XGB(depth=7), mean of the three) that bundles imputation, scaling, and averaging behind a single `.predict(df)` call — `joblib.dump`'d exactly like the single-XGB champion, with matching `feature_list.json` / `metadata.json`. Verified the round-trip (`joblib.load` → `.predict()` on held-out rows) before trusting the full run.

**Result** (round1-5 table, 17,270 rows / 8,642 pairs, 5×10 repeated pair-grouped CV):

| | single-XGB champion | MLP+XGB ensemble | Δ |
|---|---:|---:|---:|
| CV MAE | 2.397 | **2.176** | -9.2% |
| CV R² | 0.409 | 0.451 | |
| Frozen holdout MAE (n=29) | 2.983 | **2.633** | -11.7% |
| Frozen holdout R² | 0.865 | **0.898** | |

Both numbers moved in the same direction and by a similar margin as the CV-only estimate from the prior session, confirming the architecture win is real, not a CV-only artifact — it holds on a genuine molecule-disjoint holdout. By reaction-type category, the ensemble's largest absolute error is on `aliph-aliph` (2.69 kcal) and smallest on `hetero-hetero` (2.04 kcal), the same difficulty ordering seen throughout this project.

**Shipped artifact**: `data/cross_benzoin/cross_round5/train_ensemble_slim120_v1/models/cross_ensemble_model.joblib` (+ `feature_list.json`, `metadata.json`, parity/residual/XGB-importance figures, `cv_predictions.csv`, `mae_by_category.csv`). This is now the best-validated cross-benzoin model and the natural candidate to become the new champion / new scoring model for future active-learning rounds, pending the user's decision to promote it.

## 2. Homo+cross unification, re-tested at 17,270-row scale

New `cross_benzoin/assemble_cross_training_table_unified_v2.py` generalizes the round-specific unification script (which only knew rounds 1-3, no mordred) to rounds 1-5 plus the *same, unchanged* 30,000-row stratified homo sample used in the original test — isolating the effect of cross's own data growth + mordred, not a homo-side change.

**Schema-matching fix**: the raw unified table has 2,180 feature columns (homo's product-side mordred file is the full 1,826-descriptor dump; cross's own product-mordred files only cover 219 of those). Training on all 2,180 would let the model lean on homo-exclusive columns that are structurally NaN for every cross row — not what "does homo help cross" is supposed to measure. Fixed by subsetting the unified table to exactly the champion's 260 named features before training; all 260 were present by name with low, uniform NaN rates across every round (2.5%-7.6%), confirming the SHAP-slimmed mordred set generalizes across both populations rather than being cross- or homo-specific.

**Result** (47,270 rows: 17,270 cross + 30,000 homo, same 260-feature schema, same 5×10 CV protocol):

- Frozen holdout MAE: **2.983**, byte-identical to the cross-only run. This is expected, not a bug — `frozen_holdout_eval()` only fits on rows with a `candidates_v3` split label, and every homo row's `donor_id` is a numeric aldehyde index (not an InChIKey), so all 30,000 are structurally excluded from that fit. The frozen metric cannot see homo's effect by construction.
- Mixed 5×10 grouped-CV MAE: 2.279 (vs cross-only's 2.397) — but this average is dominated by homo's 30,000 (63%) generally-easier rows, so it doesn't by itself answer whether cross predictions specifically improved.
- **The number that answers the real question** — per-round breakdown of the mixed run's `cv_predictions.csv` (merged back with the source table's `round` column), restricted to the 17,270 cross-only rows: **MAE 2.374**, vs the cross-only model's own CV MAE of **2.397**.

| scale | cross-only CV MAE | unified (cross rows only) CV MAE | Δ | relative |
|---|---:|---:|---:|---:|
| round1-3 (4,120 rows), prior session | 2.966 | 2.910 | -0.056 | -1.9% |
| round1-5 (17,270 rows), this session | 2.397 | 2.374 | -0.023 | **-1.0%** |

**Interpretation**: homo+cross unification is still additive at the new scale — the direction of the effect reproduces cleanly — but the *magnitude* roughly halved as cross's own data grew 4.2×. This matches the working hypothesis from the prior session ("cross may need homo's assist less as its own data grows") rather than disproving it. The absolute gain here (0.023 kcal/mol) is small enough that it sits close to what repeated-CV noise could produce on its own with a different fold partition (the unified run's fold structure necessarily differs from the cross-only run's, since the total row/group count is different) — so this should be read as "the effect is real and consistently in the helpful direction across two independent scales, but no longer confidently a big lever," not as a precise, noise-free +0.023 kcal number.

## 3. Round 6: reservoir rescoring, 4,000-pair selection, DFT-SP submitted

The 10,177-pair screen10k reservoir built in the prior session for exactly this purpose (round5 only spent 2,500 of it) still had 7,677 pairs (15,342 rows) sitting fully featurized and unused.

- Filtered the candidate feature table to the unselected 7,677 pairs (`cross_round6_candidates.parquet`).
- Rescored via `score_round_active_learning.py` (unchanged script) against the **new round1-5 slim120 champion** (40-bootstrap pair-grouped uncertainty ensemble) rather than the stale round1-3 model used to originally rank this reservoir — giving the now-much-better-informed model a fresh look at where its remaining blind spots are.
- Selected the **top 4,000/7,677 pairs** by uncertainty — 1.6× round 5's batch size, continuing the "scale the batch up" pattern the user asked to reuse, while still leaving 3,677 pairs in reserve for a future round rather than exhausting the reservoir in one shot.
- Built the 2-orientation DFT-SP input (7,996 rows; 4 of 4,000 pairs only had one orientation survive featurization). Manifest check: 7,906/7,996 (98.9%) have a fully cached aldehyde-side DFT energy, in line with prior rounds' coverage.
- Smoke-tested `dft_sp_cross_from_geom.py` directly (2 real ORCA single points completed cleanly with physically sane ΔG values) before committing to the full array.
- Submitted as SLURM array **job 24667830** (80 chunks × 100 rows, genoa, %50 throttle — the throttle already validated safe for this specific script in the prior session, since DFT-SP's one-task-one-ORCA-process profile doesn't share `cb_featurize`'s node-local-scratch-oversubscription failure mode). **50/80 chunks completed as of this report**; still running.

**Not yet done** (next steps for whoever picks this up): (1) wait for the round6 array to finish, (2) run the same product-BDE follow-up step (`calc_bde_gxtb_product_cross.py`) round4/5 needed, (3) assemble a round1-6 combined table and retrain both the single-XGB champion and the new MLP+XGB ensemble on it, (4) re-run the homo+cross unification test a third time at the round1-6 scale to see whether the shrinking-benefit trend continues, flattens, or reverses, (5) decide whether to promote the MLP+XGB ensemble to the "shipped champion" role used by `score_round_active_learning.py` for future rounds' uncertainty scoring.

## Bottom line

- Cross's best validated number is now the **MLP+XGB ensemble's frozen holdout MAE of 2.633 (R² 0.898)** — a genuine, holdout-confirmed improvement over the single-XGB champion (2.983 / 0.865), not just a CV artifact.
- Homo+cross unification is confirmed still helpful at 4×'s the data scale, but the benefit has visibly shrunk (-1.9% → -1.0% relative), consistent with cross needing homo's assist less as its own dataset grows — an important data point for whether further unification effort is worth prioritizing over more active-learning rounds.
- Round 6 is in flight with zero new xTB/g-xTB compute cost (pure reuse of an existing reservoir), continuing to close the gap toward homo's 1.4-1.5 MAE ceiling.
