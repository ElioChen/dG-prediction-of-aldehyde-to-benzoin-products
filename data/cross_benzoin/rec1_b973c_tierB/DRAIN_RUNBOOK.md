# Tier B drain → champion+GNN retrain runbook

Prepared 2026-09-08 (deep-night session) so drain execution (HANDOFF_20260908 §1.3) is
mechanical. Campaign monitor `b7tmy1z7q` emits `*** Tier B ALL 3 ARMS DRAINED ***` when
`squeue` shows 0 tasks for `26467946 26467948 26468648`.

## Key facts verified this session

- Tier B `pid` == champion table `id` **exactly** (35,528/35,528, string InChIKeys, no
  `norm_id` needed). `new_scaffold_split` matches 100%; stored `label`==`dG_orca_kcal`,
  `gxtb_stored`==`dG_gxtb_kcal` (max|Δ|=0).
- Master list `rec1_b973c_tierB_pairs_35528.csv` is **front-loaded**: all 481 validation +
  all 448 test pairs are in the first ~3.5k rows → an early partial merge already has the
  full scaffold-disjoint holdout.
- 257 feature list = 260 minus exactly `{donor,acceptor,product}_n_CHO`
  (`data/cross_benzoin/feature_list_257_no_nCHO_v2.json`).
- `train_cross_delta.py` `TARGET_COL`/`BASELINE_COL` are now env-overridable
  (`CB_TARGET_COL` / `CB_BASELINE_COL`, defaults unchanged). `train_scaffold_disjoint.py`
  and `train_cross_gnn_arch_sweep.py` both import those names → one env covers all three.
- GNN trainer reads its feature list from `<champion-dir>/models/feature_list.json`, so
  pointing `--champion-dir` at the new 257 retrain dir gives it the 257 schema for free.

## Steps at drain

```bash
cd /gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
GPY=/home/schen3/venv/nequip/bin/python
TB=data/cross_benzoin/rec1_b973c_tierB
R10=data/cross_benzoin/cross_round10
NEW=$R10/scaffold_disjoint_10rounds_b973c_v1
```

**1. merge shards + QC**
```bash
$PY cross_benzoin/merge_rec1_b973c_tierB.py
#   -> $TB/rec1_b973c_tierB_{merged,labels,summary}.csv/.json
#   READ $TB/rec1_b973c_tierB_summary.json: coverage should be ~35,528 (few true-fails
#   ok); "verdict" GREEN/AMBER = lever survived. If RED, STOP and reassess before retrain.
```
Requeue any true-fail shards first (col6 `dG_r2scan_kcal` empty; not a `\r` artifact):
`sbatch --partition=rome --array=<failed ids>%128 pipeline/slurm/submit_rec1_b973c_tierB.sh`

**2. build the slim257 + B97-3c training table**
```bash
$PY cross_benzoin/patch_train_table_tierB_b973c.py \
    --table $R10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet \
    --labels $TB/rec1_b973c_tierB_labels.csv \
    --out $R10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet \
    --keep-unrelabeled drop --winsor
#   review the printed per-column means/std + resid(r2scan-b973c) std (~0.9-1.1 expected).
#   --winsor clips the audit's named heavy-tail feats (wbo_CC_new, mulliken_carbC,
#   mulliken_CHO_C) to clean-train [p1,p99]. Per the audit: FIRST re-run
#   cross_benzoin/feature_audit* on the regenerated geoms; only keep --winsor if the
#   same pairs still tail. Drop it for an A/B.
```

**3. champion retrain (CPU, ~1 h; use SLURM not login node for the full repeats=10)**
```bash
CB_TARGET_COL=dG_r2scan_kcal CB_BASELINE_COL=dG_b973c_kcal \
  $PY cross_benzoin/train_scaffold_disjoint.py \
    --table $R10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet \
    --outdir $NEW --folds 5 --repeats 10 --seed 42
#   -> $NEW/models/{champion,ensemble}_scaffold_disjoint.joblib + feature_list.json (257)
#   report: scaffold_disjoint_holdout_xgb.MAE / _ensemble.MAE (target ~1.0-1.5 = breakthrough)
```

**4. GNN 4-seed retrain** — copy `submit_gnn_seed_ensemble_r10_cpu.sh` → `_b973c.sh`,
swap `--table` to the slim257_b973c parquet, `--champion-dir`/`--ensemble-path` to `$NEW`,
`OUT=$R10/gnn_attentive_10rounds_b973c_seed${SEED}`, and add to the env block:
`export CB_TARGET_COL=dG_r2scan_kcal CB_BASELINE_COL=dG_b973c_kcal`. Keep `--array=1-4`.
`sbatch --partition=<idle rome|fat_rome> cross_benzoin/slurm/submit_gnn_seed_ensemble_r10_b973c_cpu.sh`

**5. blend eval** — `blend_gnn_seed_ensemble_r10.py` (via `submit_blend_gnn_seed_ensemble_r10.sh`,
genoa short-borrow ~10 min), pointed at the 4 new seed dirs + `$NEW`. Expect best
`w_gnn ≈ 0.75` (this session's recipe). New blend MAE = the headline.

**6. if holdout ensemble/blend MAE lands ~1.0–1.5** → project-level breakthrough:
- rewrite `cross_benzoin/CHAMPION.md` (new champion = r1-10-b973c), `PROJECT_SUMMARY_20260907*.md`
- `sbatch cross_benzoin/slurm/submit_backup_recovery_artifacts.sh`
- update memory `cross-r1-10-champion-and-label-ceiling`

**7. launch the homo full-library relabel campaign** (user-directed 2026-09-10;
infrastructure prepared this session — full plan
`data/cross_benzoin/homo_standalone/CAMPAIGN_PLAN.md`). Independent of steps 1-6
(only shares cluster capacity) — start once Tier B's arms have freed their slots:
```bash
# smoke first if not already green (job 26538271):
#   check data/cross_benzoin/homo_standalone/relabel/smoke/smoke.csv has 2 rows, dG_r2scan filled
# then, multi-arm (mirror Tier B's rome QOS-128 + fat_genoa split):
sbatch --partition=rome      --array=0-6734%128    cross_benzoin/slurm/submit_homo_relabel.sh
sbatch --partition=fat_genoa --array=6735-13469%150 cross_benzoin/slurm/submit_homo_relabel.sh
# build a campaign monitor on data/cross_benzoin/homo_standalone/relabel/shards/*.done (/13470)
# Phase 2/3 (merge + assemble + single-XGB + single-GNN) = post-drain of THIS campaign,
# ~3 weeks out; see CAMPAIGN_PLAN.md.
```

## Rollback

Nothing here mutates the shipped champion. New artifacts land in `*_b973c_v1` dirs and a
new parquet. The `CB_*` env defaults are the old columns, so every other caller of
`train_cross_delta` is unaffected.
