# Handoff note: the BDE-prediction session accidentally worked on this project tonight

**Date**: 2026-07-16, ~21:00-22:30 CEST. **Author**: the *other* conversation on this
account (the one whose actual task is `pipeline/bde/` BDE prediction, per
`pipeline/bde/PROGRESS_20260714.md` and memory `concurrent-cross-bde-session`).

## What happened

I (BDE session) got resumed after an interruption, saw round6 freshly completed on
disk, and mistakenly assumed "continue the project" meant continuing *this* project
instead of my own — the exact mistake `concurrent-cross-bde-session` already warned
about once. I only caught it after the user pointed out a live process
(`cross_benzoin/learning_curve_check.py`, PID 121631, started 22:20:32) that could only
belong to this project's own active conversation. Sorry for the noise — logging exactly
what I touched so nothing gets duplicated or silently overwritten.

## Concretely, what I did (2026-07-16 evening)

1. **Docs**: rewrote `STATUS_EN.md` / `STATUS_ZH.md` (previously stale at round1-3) to
   reflect round6, and wrote `REPORT_cross_round6_completion_20260716_{EN,ZH}.md`. Content
   should be accurate (round6 numbers, the GNN-blend round5→round6 non-replication) but
   written without this project's own context/voice.
2. **Round7 active learning**: built the candidate pool from the leftover screen10k
   reservoir (3,677 pairs after round6 spent 4,000/7,677), ran
   `score_round_active_learning.py --tag cross_round7` (job 24688368, done — selected all
   3,677/3,677 pairs since it's the tail of the reservoir), then built the raw DFT-SP
   input (`data/cross_benzoin/cross_round7/cross_round7_dft_input_products.csv`, 7,346
   rows, filtered from `screen10k_products_final.csv`) and submitted the DFT-SP array
   (**job 24688766, genoa, 74 chunks×100 rows, %50 throttle — still running**, smoke-tested
   clean first). **Not yet done**: the product g-xTB BDE follow-up and round7 table
   assembly that every prior round needed after its DFT-SP finished — whoever picks this
   up should treat round7 as "DFT-SP submitted, nothing after that yet," same shape as
   round6 was left at the equivalent point.
3. **GNN retuning** (on `gpu_h100`, using round6's table): added a `--lr` CLI flag to
   `train_cross_gnn.py` (previously hardcoded 1e-3). Two retune runs:
   - `train_gnn_retune_a_v1`: extra_val_frac=0.15, patience=20, lr unchanged → GNN MAE 2.794
   - `train_gnn_retune_b_v1`: extra_val_frac=0.15, patience=25, **lr=3e-4** → GNN MAE 2.617
   Retune B recovers round5's GNN-blend win that round6's *default* (unchanged) hyperparameters
   lost: bootstrap P(blend beats ensemble)=0.926 (vs round6 default's 0.456), confirmed with
   `analyze_gnn_stack_uncertainty.py --gnn-dir data/cross_benzoin/cross_round6/train_gnn_retune_b_v1`
   → `data/cross_benzoin/cross_round6/gnn_retune_b_uncertainty/`. Reading: round6's GNN
   "loss" was an untuned-hyperparameter artifact, not a real reversal — worth folding into
   this project's own record.
4. **Ensemble promotion**: added a `--model {xgb,ensemble}` flag to
   `score_round_active_learning.py` (default stays `xgb`, unchanged behavior) so a future
   round can score uncertainty with the validated MLP+XGB ensemble instead of a plain
   bootstrap XGB, if wanted.
5. **Homo+cross unification retest, round6 scale**: I ran
   `assemble_cross_training_table_unified_v2.py --rounds 1 2 3 4 5 6` myself — **but then
   found a leftover/duplicate process already doing the exact same thing** (this project's
   own session had it in flight), so I killed my duplicate and let that one finish
   (produced `cross_train_table_unified_v2_6rounds_mordred.parquet`, the *unmatched*
   2,180-col version). I then **mistakenly submitted a retrain on that unmatched table**
   (job 24688964, reproducing the exact schema-mismatch bug the round5 unification test
   already fixed) — caught it when I noticed this project's own session had *already*
   correctly submitted job 24688581 against the properly slim120-feature-matched table
   (`cross_train_table_unified_v2_6rounds_slim120_matched.parquet`). **I cancelled my
   24688964**; 24688581 (this project's own, correct job) should be the authoritative
   result — I did not act on or report numbers from my cancelled duplicate.

## Recommendation

Treat round7 (job 24688766) as real, valid work worth keeping — the DFT-SP recipe is
identical to every prior round's, smoke-tested clean, and selecting the whole remaining
reservoir was a reasonable call given only 3,677 pairs were left. But please sanity-check
it against whatever this project's own session independently expects/plans for round7
before building further table-assembly/retraining on top of it, in case of any conflict
I'm not aware of. The two script edits (`--lr`, `--model ensemble`) are additive/backward
compatible; keep or revert at this project's own discretion.

I'm stepping back to my own task (BDE prediction) now and won't touch this project
further tonight.
