# Cross-benzoin Δ-learning: living project status

_A continuously-updated overview — unlike the dated `REPORT_*` files (point-in-time
snapshots), this file gets edited in place as things change. Companion:
`STATUS_ZH.md`. Last updated: 2026-07-16 (caveat + round8 sections added 2026-07-17)._

## ⚠ Correction added 2026-07-17: every frozen-holdout MAE number below (†) was measured on a split that leaks scaffolds — treat as optimistic

Every "frozen holdout MAE" number in this document and in this project's round-by-round
history (2.966 → 3.043 → 2.983 → 3.132 / 2.633 → 2.582, etc., all marked with a † below) was
measured on `candidates_v3`'s *original* molecule-level (InChIKey-disjoint) train/test split.
A follow-up diagnostic found that split leaks Bemis–Murcko **scaffolds** severely even though
it is molecule-disjoint: **93% of the old 29-row frozen holdout had at least one side's
(donor or acceptor) scaffold already present in train** (only 2/29 pairs were genuinely
scaffold-novel), and an independent, well-powered scaffold-disjoint CV test (n=2255, 5
repeats) found a real, reproducible **+0.221 MAE (~9.8% relative) generalization gap** between
interpolation and genuinely novel scaffolds. Net effect: **every historical
frozen-holdout number in this project overstates true novel-chemistry generalization by
roughly 0.2–0.5 MAE** — don't read them as tight upper bounds on real-world error. Full
detail: memory `cross-five-diagnostics-20260717`.

**This has been fixed going forward.** `candidates_v3` was rebuilt with a genuinely
scaffold-disjoint split (`rebuild_scaffold_disjoint_split.py`, one global greedy
largest-scaffold-first bin-packing pass, verified zero scaffold overlap), round1-7's labeled
data was re-split against it (clean-train=19,687, **clean-test=450** — vs the old n=29, a
~15.5x larger and much more statistically trustworthy held-out set purely by size), and both
architectures were retrained on it. New headline numbers on the honest split: single-XGB MAE
2.448/R² 0.730, MLP+XGB ensemble MAE 2.256/R² 0.763, best GNN+ensemble blend MAE **2.215**
(bootstrap-confirmed real: P(blend beats ensemble)=0.9632, 90% CI on the gain
(-0.080, -0.003), B=20,000). **Caveat on these new numbers too**: naively comparing them
against the old † numbers looks like an improvement, but that's a composition artifact — the
new 450-row test set happens to skew toward easier reaction-type categories (hetero-heavy).
Reweighted to a representative category mix, the honest, apples-to-apples cost of removing the
leak is a real but modest **+0.14 to +0.30 MAE** versus the old (still-leaky) CV numbers,
consistent with the diagnostic's independent +0.221 estimate. Full detail: memory
`cross-scaffold-disjoint-rebuild-20260717`.

**Going forward, use `data/cross_benzoin/cross_round7/
cross_train_table_7rounds_scaffold_split_labeled.parquet` (`new_scaffold_split` column) as the
evaluation resource, and `cross_benzoin/predict_cross_champion.py`
(`CrossBenzoinBlendPredictor`) as the recommended production model** — not the old
candidates_v3-split-based frozen-holdout / champion artifacts referenced below (kept for
historical continuity, per this project's preserve-history convention; every number carried
forward from before this correction is marked with a † the first time it appears in each
section).

## Where we are

**Goal**: predict benzoin-condensation ΔG (kcal/mol) for an arbitrary pair of
aldehydes (A+B, A possibly == B) cheaply and accurately, via Δ-learning:
`dG_pred = dG_gxtb_baseline + ML_correction`. Two model lines exist:

| | homo (A+A) | cross (A+B, A≠B) |
|---|---|---|
| status | **shipped, validated** | **6 active-learning rounds done, validated** |
| label coverage | 219,364 DFT labels (full 220,859-aldehyde library) | 12,597 pairs (25,176 directed rows) |
| best-validated artifact | `pipeline/models/gxtb_dft_correction_ENSEMBLE72_20260626.joblib` (test MAE 1.503) | `data/cross_benzoin/cross_round6/train_ensemble_6rounds_slim120_v1/models/cross_ensemble_model.joblib` (frozen-holdout MAE 2.582†, R² 0.897) — superseded by the scaffold-disjoint blend MAE 2.215, see caveat above |
| inference entry point | `benzoin_dG.predict_dG_champion()` | not yet packaged as an installable API, and not yet promoted to be the scoring model future active-learning rounds use (still using the single-XGB champion line, see below) |

**These are currently two separate model lines**, not one general model. Unification
has been tested twice (Open question 1) and helps, but the benefit is shrinking as
cross's own data grows — see below.

## Cross-benzoin active-learning loop: 6 rounds done

Frozen holdout is always the same 29 molecule-disjoint `candidates_v3` test rows;
the g-xTB baseline MAE on those rows (5.315) is therefore identical at every round —
a standing sanity check that the round-to-round comparison stays apples-to-apples.
**† All `frozen *` rows in the table below are on the old, scaffold-leaky split — see the
correction note at the top of this document; the honest scaffold-disjoint replacement
(n_test=450) gives single-XGB MAE 2.448/ensemble 2.256/blend 2.215.**

| | R1-3 | R1-4 | R1-5 | R1-6 |
|---|---:|---:|---:|---:|
| rows / pairs | 4,120 / 2,062 | 12,378 / 6,194 | 17,270 / 8,642 | 25,176 / 12,597 |
| CV champion MAE | 2.966 | 2.261 | 2.397 | 2.276 |
| CV champion R² | 0.750 | 0.751 | 0.409 | 0.473 |
| CV g-xTB baseline MAE | 5.335 | 4.191 | 4.499 | 4.292 |
| frozen champion MAE † | 3.398 | 3.043 | 2.983 | 3.132 |
| frozen champion R² † | 0.842 | 0.859 | 0.865 | 0.854 |
| frozen ensemble MAE † | — | — | 2.633 | 2.582 (best so far on this split) |
| frozen ensemble R² † | — | — | 0.898 | 0.897 |

Sampling method: round 1 = category-diversity; rounds 2-6 = bootstrap-ensemble
uncertainty active learning (round 6 rescored the leftover screen10k reservoir
against the round1-5 champion before selecting the top 4,000/7,677 pairs by
uncertainty). Full details in the dated reports, most recently
`REPORT_cross_round6_completion_20260716_{EN,ZH}.md`.

**`n_test` is structurally capped at 29** under the current recipe: every round
samples only from `candidates_v3`'s `train` split (correctly — you never want a new
training round touching held-out molecules), so the test set can only grow if a
future round *deliberately* draws some `test`-split pairs *for evaluation only,
never training*. Not done yet; worth deciding on purpose at some point, since n=29
gives fairly wide confidence intervals on the reported R²/MAE — see the GNN-blend
non-replication finding below for a concrete example of why that caution matters.

**Two architecture lines beyond the single-XGB champion, both validated on a frozen
holdout**:
- **MLP+XGB ensemble** (`train_cross_ensemble.py`): StandardScaler + MLP(128,64) +
  XGB(depth=5) + XGB(depth=7), averaged. Consistently the best model at every scale
  tested (round5: 2.633†, round6: 2.582†) — the current best-validated artifact on
  the old split, but **not yet promoted** to be the scoring model
  `score_round_active_learning.py` uses for future rounds (that still uses the
  single-XGB line). See the round8-prep section below for the pool this scoring
  script should now draw from.
- **Triple-encoder GNN** (`train_cross_gnn.py`, homo-pretrained, GINE atom/bond
  featurization + QM-scalar readout, mirrors the homo project's confirmed
  dual-encoder win): at round5 scale, a fixed 50/50 blend with the ensemble beat the
  ensemble alone with bootstrap significance (P=0.987, n=29). **This did NOT
  replicate at round6 scale** (P=0.456, no longer significant; GNN-only actually
  underperformed the ensemble, P(GNN better)=0.096) — the GNN was retrained with
  identical, untuned hyperparameters on ~46% more data and got worse in absolute
  terms (2.624→2.820) while the ensemble kept improving. Read this as "stacking is
  an open question requiring real retuning before trusting it either way," not as
  "stacking doesn't work" — a single untuned retrain isn't a fair test either.
  Details: `REPORT_cross_round6_completion_20260716_{EN,ZH}.md` §3. **Later update
  (post round6, both on the still-then-leaky split and then re-confirmed on the
  honest scaffold-disjoint split)**: this null result turned out to be an
  untuned-hyperparameter artifact — with `lr=3e-4, patience=25` the blend reliably
  beats the ensemble (old split: P=0.9261, 3-seed 2.385±0.011; new scaffold-disjoint
  split, n=450: P=0.9632, blend MAE 2.215 vs ensemble 2.256, 90% CI on the gain
  (-0.080,-0.003)). See the correction note at the top of this document and memory
  `cross-scaffold-disjoint-rebuild-20260717`.

## Open question 1 — unify homo + cross into one general model? Confirmed helpful, shrinking benefit

Tested at two scales so far, same 260-feature schema, same protocol both times:

| scale | cross-only CV MAE | unified (cross rows only) CV MAE | Δ | relative |
|---|---:|---:|---:|---:|
| round1-3 (4,120 rows) | 2.966 | 2.910 | -0.056 | -1.9% |
| round1-5 (17,270 rows) | 2.397 | 2.374 | -0.023 | -1.0% |

**Reading**: unification is real and consistently helpful in direction, but the
magnitude roughly halved as cross's own data grew 4.2× — consistent with cross
needing homo's assist less as its own dataset grows. **Round6 (25,176 rows) is the
natural next data point** to see whether this trend flattens, keeps shrinking, or
reverses — not yet re-tested at this scale. `assemble_cross_training_table_unified_v2.py`
already generalizes to any round count, so this is a cheap re-run (no new compute),
just needs pointing at the round6 table.

## Open question 2 — is a ~10k medium-scale cross-product campaign worth it?

Superseded by events: rounds 4-6 already added ~21,000 rows via targeted active
learning (not a blind 10k scale-up), which is exactly the more-justified path this
question's answer recommended. The original learning-curve plateau observation
(round1-3 CV MAE flat 2.93-2.97 from 50%-100% of that round's data) held at the time
but the project's own subsequent history (round4's CV MAE dropping to 2.261) shows
targeted active learning at a new scale can still move the number meaningfully —
plateaus observed within one round's data don't necessarily predict the next round.

## Known small gaps (not urgent)

- `interaction_*` feature block (10 features) confirmed useless across every round
  tested (n=598 through 25,176) — safe to drop in a future cleanup, harmless to keep.
- ~0.15%-of-library aldehyde DFT-SP timeout gap (documented long ago, memory
  `dftsp-timeout-3600-snapshot`) still occasionally costs a handful of rows per
  round; not worth a dedicated backfill at current scale.
- Product BDE compute success rate holds steady at ~94% across all rounds
  (geometry/bond-detection failures on some ring systems); acceptable, no action
  needed. Round6's raw DFT-SP step itself had 0/7,996 errors.

## Round 8 candidate batch prepped (2026-07-17) — awaiting user go-ahead to launch compute

Round7 completed (see the correction note at the top of this document): dataset now
32,456 rows / 16,241 pairs, ensemble CV MAE 1.883 (best yet, on the old split).
Screen10k's reservoir (used for rounds 2-7's uncertainty sampling) is now fully
exhausted (round5 2,500 + round6 4,000 + round7 3,677 = 10,177/10,177 spent).

Per the scaffold-disjoint rebuild's open item #4 (memory
`cross-scaffold-disjoint-rebuild-20260717`), round8's candidate pool must be
filtered through the new scaffold-split lookup table
(`data/cross_benzoin/candidates_v3/candidates_v3_pairs_with_scaffold_split.parquet`)
before any uncertainty scoring, so future rounds don't leak scaffolds into the new
honest scaffold-test/validation sets the way rounds 1-7 leaked into the old
molecule-level frozen holdout.

- **Filtered eligible pool**: 1,263,002 pairs that are both (a) on the
  `new_scaffold_split == "train"` side (never `test`/`validation`/`mixed`) and (b)
  not already labeled by round1-7 (cross-checked: 15,063/16,241 of round1-7's pairs
  are inside candidates_v3 proper and correctly excluded; the other 1,178 came from
  outside sources like screen10k and are structurally absent from this table, as
  expected).
- **Round8 candidate batch**: `cross_benzoin/sample_round8_from_candidates_v3.py`
  drew a category-balanced sample (667 pairs × 6 class_pair combinations = 4,002
  unordered pairs / 8,004 directed rows, seed 42) from that eligible pool, matching
  round6's batch size (the largest single targeted-draw round to date). Written to
  `data/cross_benzoin/cross_round8/cross_round8_pairs.csv`. **This selection step
  used no compute** — it's a pandas join + stratified sample against existing
  parquet/CSV tables, no xTB/g-xTB/DFT calls.
- **NOT done, needs explicit approval**: real uncertainty-based *scoring* (as
  opposed to this pre-selection) requires the pool to be featurized first
  (`cb_featurize.py` fuses xTB conformer search + g-xTB SP — genuine compute), which
  `score_round_active_learning.py` then consumes. That featurization was
  deliberately NOT launched this session. Exact next commands, in order, once
  approved:

  1. **Featurize** (xTB + g-xTB compute, genoa array, reuses the cached 220,859-aldehyde
     library so only the product side is new compute):
     ```
     IN=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/cross_round8/cross_round8_pairs.csv
     OUT=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/cross_round8
     N=$(($(wc -l < "$IN")-1)); CHUNK=100; NCH=$(( (N+CHUNK-1)/CHUNK )); mkdir -p "$OUT/logs"
     sbatch --array=0-$((NCH-1))%64 --output="$OUT/logs/cb_%A_%a.out" \
       --export=ALL,INPUT="$IN",OUTDIR="$OUT",CHUNK=$CHUNK,\
     ALD_CACHE=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/aldehydes_all.csv,\
     REQUIRE_CACHE_COMPLETE=1 cross_benzoin/slurm/submit_cb_featurize_array.sh
     ```
  2. **Merge chunk outputs** into one `cross_round8_products.csv`, then **product BDE**
     (`pipeline/compute/calc_bde_gxtb_product_cross.py --products-csv ... --out-dir
     data/cross_benzoin/cross_round8/bde_gxtb`, array over chunks, same recipe as
     round2/round6's BDE follow-up jobs).
  3. **Assemble features**: `python cross_benzoin/assemble_cross_round_features.py --tag cross_round8`.
  4. **Uncertainty-score** against the corrected, scaffold-clean training pool (not
     the old leaky one — see the round7-scaffold-check precedent,
     `slurm/submit_score_round7_scaffold_check.sh`):
     ```
     python cross_benzoin/score_round_active_learning.py --tag cross_round8 \
         --train-table data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_clean_train.parquet \
         --feature-list data/cross_benzoin/cross_round7/scaffold_disjoint_v1/models/feature_list.json \
         --model ensemble --n-boot 40 --n-select 4000 --seed 42
     ```
  5. **DFT-SP** on the top-uncertainty selection via
     `pipeline/compute/dft_sp_cross_from_geom.py` (r2SCAN-3c, same recipe as every
     prior round's labeling step) — only after steps 1-4 and an explicit go-ahead.

## Round 8 compute IN FLIGHT (2026-07-17, user-approved — idle genoa capacity)

User approved launching the round8 pipeline (84 idle genoa nodes confirmed via
`sinfo -p genoa` at approval time). Executing the exact 5-step sequence above.

- **Step 1 (featurize) LAUNCHED**: job **24707568**, `cb_feat` array,
  `--array=0-80%64` (81 chunks × 100 pairs = 8,004 rows), genoa, using
  `ALD_CACHE=data/cross_benzoin/homo_v6/aldehydes_all.csv` +
  `REQUIRE_CACHE_COMPLETE=1` so only product-side xTB+g-xTB compute runs (donor/
  acceptor aldehydes are all in the cached 220,859-library). Verified healthy at
  launch: 20+ tasks RUNNING within 20s, `chunk 0: pairs 0:100` sliced correctly, no
  immediate errors. Output: `data/cross_benzoin/cross_round8/chunk_NNNN/`. Historical
  per-chunk runtime for this step is ~1-2h (occasionally ~1min when a chunk is
  entirely cache-hits), so with %64 throttle over 81 chunks expect roughly 2-4h
  wall-clock for the array to fully drain.
- Steps 2-5 (product BDE, assemble, score, DFT-SP) not yet started — pending step 1.
  This section will be updated as each lands.

- **Promote the round6 MLP+XGB ensemble** to be the shipped champion / the scoring
  model `score_round_active_learning.py` uses for future rounds, since it's now the
  best-validated artifact at every scale tried.
- **Retune the GNN** (learning rate / epochs / capacity) at the round6 scale before
  drawing any conclusion about whether stacking is worth pursuing — the round5
  "win" and round6 "loss" are both single, untuned data points.
- **Round7**: 3,677/7,677 pairs still sit unused in the screen10k reservoir after
  round6 spent 4,000 of them — zero new xTB/g-xTB compute needed to draw from it.
- **Re-test homo+cross unification a third time** at the round6 (25,176-row) scale
  using the already-generalized `assemble_cross_training_table_unified_v2.py`.

## 2026-07-20 addendum: split-ratio decision finalized

The user flagged that the pair-level split ratio (60.7% train / 1.4% val / 1.5% test / 36.8%
mixed, a consequence of the molecule-level 80/10/10 target being squared by the "both sides
must match" derivation rule) looked unreasonable and asked for a 7:2:1 evaluation. Both
options (molecule-level 70/20/10 vs 80/10/10) were fully retrained end-to-end
(champion+ensemble+GNN) for comparison:

| | 80/10/10 (production) | 70/20/10 (archived reference) |
|---|---:|---:|
| clean train rows | 19,687 | 14,874 |
| blend MAE (honest holdout n=450) | **2.215** | 2.614 |
| bootstrap 90% CI (blend vs ensemble) | (-0.080, -0.003), excludes zero | (-0.001, 0.100), barely includes zero |

User's final decision: **80/10/10 is production** (`scaffold_disjoint_v1`,
`predict_cross_champion.py`'s default), since it has more training data, better accuracy, and
a cleaner significance test — and the larger validation bucket that 70/20/10 buys only
benefits the GNN's early-stopping (not the tabular models at all), and didn't show up as a
real accuracy edge in practice. The 70/20/10 artifacts (`scaffold_disjoint_721_v1`) are kept
in full as a validated archived reference, not deleted. Both versions of the round9+ eligible
pool now exist (`candidates_v3_pairs_with_scaffold_split.parquet` = 80/10/10,
`_721.parquet` = 70/20/10) — use the 80/10/10 version going forward.

## 2026-07-20 addendum: the documented 5-step pipeline was missing a step, fixed at the script level

While actually running round8, `assemble_cross_round_features.py` came back missing 53
`product_mordred_*` features (the champion's 260-feature schema has included them since
round6/7, but the documented 5-step recipe never wrote this step down). Fixed:
1. **Patched this round**: ran `add_mordred_cross_products.py` (reuses already-saved xyz
   geometry, no re-optimization needed, ~1s/molecule) and re-assembled.
2. **Fixed the script itself, not just this one round**: `assemble_cross_round_features.py`
   now auto-discovers `{rdir}/mordred_products/chunk_*.csv`; if that's absent and
   `--product-mordred-csv` wasn't passed either, it now **hard-fails** (requires an explicit
   `--allow-missing-mordred` to proceed) instead of silently writing a table that looks
   complete but is missing 53 of 260 features, as happened this time.

**Round9+'s correct 6-step pipeline** (was documented as 5):
1. Featurize (cb_featurize array, CHUNK=30, run `check_aldehyde_cache_coverage.py` first)
2. Merge chunks + product-side g-xTB BDE array
3. **Product-side mordred array** (`add_mordred_cross_products.py`, the newly-required step)
4. Assemble (`assemble_cross_round_features.py`, now auto-finds mordred or hard-fails)
5. Active-learning scoring (`score_round_active_learning.py`, using the production 80/10/10 model)
6. DFT-SP labeling
