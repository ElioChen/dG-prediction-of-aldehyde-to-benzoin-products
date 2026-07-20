# Benzoin dG + BDE Project: Progress Summary and Next-Stage Plan (2026-07-20)

> Covers both tracks: cross-benzoin ΔG prediction (main line) and BDE prediction (parallel
> track, formally consolidated into this session on 2026-07-17).

## 0. Friday's session interruption and recovery

Last Friday's (2026-07-17) session was cut off mid-stream while advancing the round8 data
pipeline. Checking today found three pieces of unfinished/unrecorded work:

1. **8/81 round8 featurize chunks had failed** — root-caused to 8 molecules genuinely present
   in round8's candidate pool but missing from the 220,525-entry aldehyde cache (a normal
   ~334-molecule gap between the 220,859-molecule library and the cache; round8 happened to
   draw 8 of them). Resubmitted with the cache-complete gate relaxed for just those 8 chunks
   (job **24762608**, all 8 tasks RUNNING).
2. **Two analysis scripts were written but never run**: `unification_check_round7.py`
   (re-test of homo+cross joint training at round1-7 scale, old split) and
   `unification_check_scaffold_disjoint.py` (the same re-test on the honest scaffold-disjoint
   split). Both launched today — the scaffold-disjoint one has finished (see Section 1); the
   round7/old-split one is still running (~50k-row GroupKFold CV on CPU, expect several more
   minutes).
3. **One real result that had already finished computing but was never written to any
   doc/memory**: the BDE project's round1-7-scale homo→cross fine-tuning re-test
   (`train_cross_finetune_gnn_bde_round7.py`) — its output sat in a log JSON, untouched. Now
   backfilled into `pipeline/bde/PROGRESS_20260714.md` (Section 2 below).

Separately, the pre-interruption session had flagged a concern about a possibly-conflicting
concurrent session running the same jobs. Verified today: **no actual conflict exists** — every
job flagged at the time (24692539, 24701670, 24701680, 24694239, etc.) completed cleanly on
2026-07-17. The current SLURM queue holds nothing but this session's new round8 retry and a
routine orphan-cleanup cron — that earlier concern has resolved itself.

## 1. Cross-benzoin (ΔG prediction) current state

| Item | Value | Source |
|---|---|---|
| Labeled data (round1-7) | **32,456 rows / 16,241 pairs** | cumulative through round7 |
| Honest scaffold-disjoint split | train 19,687 / test **450** / val 483 / mixed 11,836 (discarded) | rebuilt 2026-07-17 |
| Current champion (honest eval) | ensemble+GNN blend **MAE 2.215** (vs ensemble-only 2.256, single-XGB 2.448) | bootstrap-confirmed real, P=0.9632 |
| Deployable predictor | `cross_benzoin/predict_cross_champion.py` | takes pre-featurized 260-column input, not raw SMILES |

**Five diagnostics** (completed 2026-07-17): phosphorus is the clearest error driver (MAE 4.14
vs 2.22 baseline); uncertainty calibration is real (Spearman 0.258, monotonic across deciles);
per-**pair** A→B vs B→A asymmetry is large (median 1.24) though the aggregate is symmetric,
cause not fully explained; candidates_v3's structural diversity matches an unrestricted
full-library draw (the pool itself isn't the bottleneck); scaffold-disjoint generalization gap
is **+0.221 (~9.8%)**, real but not catastrophic.

**Newly completed today — homo+cross unification, second independent re-check, same verdict**:

| Split | Unified-train MAE | Cross-only reference MAE | Gain |
|---|---:|---:|---:|
| round1-6, old CV split (tested 07-16) | — | — | **+1.5% (harmful)** |
| round1-7, honest scaffold-disjoint split (**run today**) | **2.478** | 2.448 | **-1.2% (harmful)** |

Two independent tests — different split methodology, different data scale — agree in
direction: **stop merging homo data into the cross training pool.** This has now been
falsified twice.

**Round8 progress**: candidate-pool selection is done (a category-balanced draw of
**4,002 pairs / 8,004 rows** from 1.263M honestly-eligible candidates — the largest single
round to date); featurize (step 1) was interrupted by the 8 failed chunks above, fixed and
resubmitted today; the remaining 4 steps (product-side BDE, table assembly, uncertainty
scoring, DFT-SP labeling) have not yet run.

## 2. BDE prediction current state

| Item | Honest value (scaffold-disjoint) | Note |
|---|---|---|
| Champion B6 (D-MPNN + local 3D descriptor fusion) | aldehyde MAE **1.579**/R²0.843, product MAE **3.060**/R²0.886 | the earlier naive-split 1.104/2.076 is confirmed inflated by 43-47% |
| B4/B5 independent retrains | both show the same 35-47% degradation | confirms the leak is a property of the data/split, not one architecture |
| Architecture tweaks (attentive pooling, chemprop 2.3) | all deltas noise-level (within ±1%) under honest eval | the earlier "small real win" conclusion does not hold |

**New result backfilled today** (round1-7-scale homo→cross fine-tuning re-test, computed
2026-07-17, never previously recorded):

| | MAE | R² |
|---|---:|---:|
| zero-shot (homo only) | 6.362 | 0.537 |
| C: homo-pretrain + cross-finetune | 4.501 | 0.695 |
| A: cross-only | 4.524 | **0.703** |

**Conclusion**: at round5 scale, fine-tuning had a clear edge (R² gap 0.010-0.016); by round1-7
scale (18,025 cross training rows) that edge has essentially vanished — R² actually slightly
favors cross-only. **Practical recommendation: future BDE cross-training rounds no longer need
the two-stage homo-pretrain-then-finetune recipe; cross-only training suffices**, freeing GPU
budget for other work.

**Still open**: no deployable checkpoint has been saved for B4/B5/B6; the `envs/gnn`
environment breakage remains unfixed/undeprecated.

## 3. The core tension: homo data vastly outweighs cross — exactly why active learning exists here

The imbalance the user pointed to — plentiful homo data, scarce cross data — is the central
tension this project has been actively managing since round2. Current state:

| | Scale | Note |
|---|---:|---|
| Homo (self-condensation) | **~219k-220k pairs**, champion MAE 1.503 | full 220,859-aldehyde library, essentially one pair per molecule (donor==acceptor), fully DFT-labeled |
| Cross candidate pool (candidates_v3) | 2M pairs (4M directed) | only **~0.0082%** of the theoretical C(220859,2)≈24.4B combinatorial space |
| Cross labeled (round1-7) | **16,241 pairs / 32,456 rows** | only **~7.4%** of homo's labeled scale |
| Cross remaining eligible (scaffold-clean, unlabeled) | **1.263M pairs** | confirmed 2026-07-17, ample headroom for round8 and beyond |

In other words, **the candidate pool itself is not the bottleneck** (structural diversity
already verified to match an unrestricted full-library draw) — **DFT-labeling throughput is**.
That's exactly why uncertainty-based active learning (bootstrap-ensemble std ranking, its
calibration verified: Spearman 0.258, monotonic across deciles) has been used since round2
instead of uniform sampling: spend the limited DFT budget on the rows the model is least
certain about, for a better generalization return per labeled row.

## 4. Next-stage plan (responding to "submit a large-scale simulation, using active learning")

**Immediate, already in motion**:
1. Round8's 8 failed featurize chunks are re-running (job 24762608); once they land, the
   pipeline proceeds through steps 2-5 (product-side g-xTB BDE → table assembly →
   bootstrap-uncertainty scoring using the **honest scaffold-disjoint ensemble model**, not the
   old leaky champion → top-N selection submitted for DFT-SP). This step *is* the standard
   active-learning execution already built into the pipeline, not a new method.

**Recommended scale-up** (grounded in the already-confirmed "CPU/GPU budget is ample, do not
throttle" finding):
2. **Round8 is already the largest single round to date (4,002 pairs)**; given ample budget and
   a 1.263M-pair remaining pool, recommend growing round9+ further to **8,000-10,000 pairs per
   round** to close the cross-vs-homo labeling-scale ratio faster (currently only ~7.4% of
   homo's scale — plenty of pool and budget headroom even several rounds larger).
3. **Layer phosphorus-targeted stratification on top of pure uncertainty ranking** — the
   diagnostics found phosphorus-containing pairs are the single clearest error driver (MAE
   4.14, nearly 2x the non-risk baseline); mildly oversampling phosphorus-containing candidates
   in round9 directly targets the known largest error source instead of waiting for uncertainty
   ranking to organically surface them.
4. **Re-test homo+cross unification a third time once cross labeling reaches the ~40-50k-row
   range** — both tests so far say harmful, but the BDE track's fine-tune re-test just
   demonstrated that "as cross data grows, the homo-augmentation effect shrinks and can flip"
   is a real pattern here — worth one more cheap (pure-CPU, reuses existing data) checkpoint.
5. **BDE track**: future cross-BDE labeling rounds (8/9+) can skip the homo-pretrain step
   (Section 2's finding); redirect the freed GPU budget toward packaging B6 into a real
   deployable checkpoint (mirroring `predict_cross_champion.py`'s pattern) — currently the only
   remaining "shippable artifact" gap for that track.

**Concrete actions already taken this session**: fixed and restarted round8's stuck featurize
step; ran the two previously-skipped analysis scripts; backfilled one already-computed but
never-recorded result (BDE round1-7 fine-tune re-test) into docs and memory; confirmed last
Friday's "concurrent session conflict" concern has resolved with no current resource
contention. Round8 steps 2-5 will continue automatically, in the existing script order, once
featurize finishes.
