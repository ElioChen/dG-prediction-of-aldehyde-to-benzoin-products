# Data-loss recovery — 2026-09-02

Around 2026-07-20 the Snellius scratch filesystem was purged. The working repo
`/gpfs/scratch1/shared/schen3/benzoin-dg` was left a hollow shell (empty directory
skeletons, an empty `.git` object store), and work stopped. On 2026-09-01 a fresh
clone of `agent/add-cross-benzoin-v2-data` was made at
`/gpfs/scratch1/shared/schen3/benzoin-dg-restored`, and a Round10 featurization was
submitted into `data/cross_benzoin/cross_round10_fat20_stage1/`.

This document records what that featurization produced, what the purge actually
destroyed, and how much of it was recovered.

## 1. Round10 stage1 featurization — complete and clean

SLURM array `cb10_fat_feat` (final id 26290821) finished 2026-09-02T01:56.

- 800 chunks x 20 pairs = **16,000 product rows**, no gaps, no duplicate ids
- **17 rows (0.11%)** with a non-empty `error`
- submitted with `EMIT_ALD=1`, so it also emitted **15,051 usable aldehyde
  descriptor rows** as a side effect

That aldehyde block turned out to matter far more than the products: it is a
drop-in substitute for the destroyed aldehyde descriptor cache, for Round10's own
pairs.

## 2. What the purge destroyed

`.gitignore` excluded `/data/raw/`, `*_all.csv`, `homo_v6/*_descriptors.csv` and
`*.parquet`, so these files existed only on scratch. Searching all three GitHub
branches' full history confirms they were never committed:

| lost | consequence |
|---|---|
| `data/raw/dft_sp_cross/*/*_dft_sp.csv` | all raw DFT-SP labels |
| `homo_v6/aldehydes_all.csv` | aldehyde xTB descriptor library |
| `homo_v6/aldehydes_mordred_slim102.csv` | 102 aldehyde-side mordred features |
| `homo_v6/aldehydes_bdfe_gxtb_descriptors.csv` | aldehyde g-xTB BDE |
| every `cross_train_table_*.parquet` | all assembled training tables |
| every GNN `.pt` | the Round9 attentive-GNN champion |

The old repo's `homo_v6/` (1,148 chunk dirs) and `data/raw/dft_sp_cross/` are empty
directory skeletons and are **not** a rebuild source.

Surviving and usable: the Round9 **tabular** champion joblibs plus
`feature_list.json` (260 features) and `metadata.json`; per-round product descriptor
tables `cross_round{2..9}_dft_products.csv`; product `bde_gxtb` chunks for rounds
2,3,4,6,7,8,9 and `screen10k` (which covers round5); product mordred for rounds
1-4, 8, 9 and `screen10k` (which covers rounds 5-7); and the 220k aldehyde library
`data/library/aldehydes_clean_v6.csv`.

## 3. Two recovery routes that worked

### 3.1 Labels survive inside CV prediction dumps

Training-run `data/cv_predictions.csv` files are tracked. In particular
`cross_round7/unification_check_v1/unified_cv_predictions.csv` holds 62,456 rows of
`id`, `round`, `dG_orca_kcal`:

```
homo_unify_v1 30000 | round4 8258 | round6 7906 | round7 7280
round5 4892 | round3 1766 | round2 1756 | round1 598
```

`load_round()` reads only `id` and `dG_orca_kcal` out of a dft_sp file, so
`cross_benzoin/synthesize_dft_sp_from_cv_predictions.py` reconstitutes the label
files and the existing assembly scripts run unmodified. **32,456 cross labels for
rounds 1-7 recovered.**

**Rounds 8 and 9 labels (~24,000 rows) have no surviving copy** and would need their
DFT single points recomputed.

### 3.2 A home-directory archive survived the purge

`/gpfs/home4/schen3/benzoin_backups/` is on home, not scratch. Its
`homo_v6_scratch_archive/` (2026-07-15) holds the per-chunk outputs that were
gitignored. Two of the three lost aldehyde files were rebuilt from it and verified
against values embedded in the tracked `cross_train_table_3rounds*.csv`:

| file | rows | agreement with original |
|---|---|---|
| `aldehydes_bdfe_gxtb_descriptors.csv` | 220,522 | max abs diff **0.000000** over 3,957 donors |
| `aldehydes_mordred_slim102.csv` | 220,524 x 102 cols | max abs diff **5.68e-14** over 4,118 donors x 102 cols |

That comparison also settled a previously undocumented convention: the aldehyde
library `id` is the **0-based row index of `data/library/aldehydes_clean_v6.csv`**
(220,859 rows, ids 0..220,858).

## 4. The one file that had to be recomputed

`aldehydes_all.csv` had no archive. It was rebuilt with a new
`cb_featurize.py --aldehydes-only` mode (the product phase costs ~6x the aldehyde
phase and every product it would recompute already survives), merging Round10
stage1's emitted aldehydes with a fresh run over the 27,303 rounds-1-7 aldehydes
stage1 did not cover.

Fidelity vs the originals, on 2,617 overlapping aldehydes
(`cross_benzoin/check_aldehyde_recompute_fidelity.py`):

- all 26 non-BDE `ALDEHYDE_FEATS`: median relative deviation **<= 0.03%**
- per molecule `G_xtb`: abs diff median **0.011 kcal/mol**, p95 1.21, p99 2.42
- ~27% of molecules show a conformer flip (>10% deviation on some feature)

That tail is intrinsic, not a bug. This project's own conformer-noise study
(`homo_v6/viz_gxtb_20260625/confnoise_summary_20260626_1559.md`) measured the spread
of dG across conformers of the *same* molecule at mean std **2.677 kcal/mol**, a
~2.14 kcal/mol single-conformer label-noise floor. The recompute-vs-original median
difference is ~200x tighter than that, i.e. as faithful as a rerun of the original
stochastic `funnel_v3` search can be.

Do not judge this check by per-feature Pearson r: near-constant features (mulliken
charges, `wbo_CO`) score low r even at excellent absolute agreement, and a handful of
conformer flips dominates the statistic.

## 5. Environment damage found along the way

The purge also emptied shared software and conda environments, and several failures
were **silent** — jobs reported COMPLETED while writing nothing or writing NaN.

- every `/gpfs/scratch1/shared/schen3/envs/*` is hollow. Use
  `/home/schen3/venv/nhc-workflow/bin/python` for tabular ML / featurization and
  `/home/schen3/venv/nequip/bin/python` for torch work.
- all 103 slurm scripts pointed `REPO` at the gutted old repo; rewritten to
  `${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}`.
- the g-xTB binary under `software/g-xtb/` is gone, but `/home/schen3/xtb/bin/xtb`
  is itself g-xTB-capable (`--gxtb` on water gives -76.435 Eh vs GFN2's -5.07 Eh).
  Confirmed consistent: Round10 stage1's `dG_gxtb_kcal` distribution (mean 2.95,
  median 2.99, p5 -3.82, p95 9.61) sits right on top of the historical rounds.
- **mordred was broken twice over.** Its package files had been deleted, and once
  reinstalled, `MoRSE.py`/`ABCIndex.py` use the `np.float` alias removed in numpy
  1.20 — `add_mordred_cross_products.py` swallows the AttributeError as NaN, so the
  entire MoRSE family came back empty while the run looked successful. Both fixed;
  219/219 descriptor columns now populate, a superset of the historical 216.
- `submit_mordred_cross_products.sh` reported COMPLETED on a missing interpreter;
  it now checks the interpreter and propagates the real exit code.

## 6. State at the time of writing

Rebuilding rounds 1-7 end to end has been validated structurally: a rounds-1-3
assembly produced a table containing **all 260 champion features** (NaN rates 3%
aldehyde mordred, 7.9% product mordred, 0.1% elsewhere) with labels and baseline
present.

The reproduction target, from the surviving 7-round run's `metadata.json`:

```
model     mlp128_64+xgb_d5+xgb_d7_ensemble
n_samples 32,456   n_pairs 16,241   n_features 260
CV(5-fold x 10 repeats)  MAE 1.8830  RMSE 3.7520  R2 0.5629
g-xTB baseline           MAE 3.9050
```

## 7. Backups

`cross_benzoin/slurm/submit_backup_recovery_artifacts.sh` archives every recovery
artifact to `/gpfs/home4/schen3/benzoin_backups/recovery_20260902/`. It is
idempotent — re-run it after any pipeline stage completes. The lesson of this
incident is narrow and concrete: **anything both gitignored and unbacked-up is one
filesystem policy change away from being gone.**
