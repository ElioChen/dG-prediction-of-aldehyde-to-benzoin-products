# Handoff: Round9 complete, Round10 decision pending

**Written**: 2026-07-21, end of the session that ran round9 fully autonomously
**Git state**: branch `agent/add-cross-benzoin-v2-data`, HEAD `32b2e0d`, all work pushed to origin
**For**: starting round10 in a fresh conversation

---

## 1. TL;DR

Round9 (16,000 candidate pairs) landed completely, round1-9 was reassembled and retrained,
and there is a **new production champion**: MAE **2.074** (was 2.106 for round1-8), confirmed
by bootstrap (P=0.988). A learning-curve check shows MAE still improving through 75% of the
current data pool (not yet plateaued) — this is evidence **for** running round10, but modest,
not a strong signal. **The round10 go/no-go decision itself has not been made** — that's why
you're reading this handoff.

If you (the user) decide to proceed with round10, Section 4 below is a ready-to-run recipe:
every script, argument, and gotcha from round9 is documented so round10 can be executed without
re-deriving anything.

---

## 2. Current champion (round1-9)

| Model | round1-8 | round1-9 |
|---|---:|---:|
| single-XGB (holdout) | MAE 2.498 / R² 0.711 | MAE 2.435 / R² 0.736 |
| MLP+XGB ensemble (holdout) | MAE 2.201 / R² 0.777 | MAE 2.163 / R² 0.786 |
| Attentive-pooling GNN (holdout) | MAE 2.168 | MAE 2.162 |
| **Best blend** (bootstrap-confirmed) | **MAE 2.106** (w=0.60, P=0.9923) | **MAE 2.074** (w=0.55, P=0.988) |

- Clean-train grew 27,583 → **43,367** rows (+57%) with round9's data.
- Held-out test set stayed **exactly n=450** across both rounds (same rows, untouched) —
  this is what makes the MAE comparison above honest/apples-to-apples.
- Bootstrap (B=20000) on the blend-vs-ensemble-only delta: 90% CI = (0.025, 0.155), entirely
  positive → the GNN blend is a real, not noise-level, improvement over the tabular ensemble
  alone, same conclusion as round1-8 (P was 0.9923 there, 0.988 here — both strong).

**Artifacts** (all under `data/cross_benzoin/cross_round9/`):
- `scaffold_disjoint_9rounds_v1/models/` — `champion_scaffold_disjoint.joblib` (single-XGB),
  `ensemble_scaffold_disjoint.joblib` (MLP+2×XGB), `feature_list.json` (the frozen 260-feature
  schema), `metadata.json`.
- `gnn_attentive_9rounds_v1/models/` — `gnn_norm_stats.joblib`, `metadata.json` (has
  `blend_w_gnn=0.55`, `best_blend_mae=2.074`). Note: `gnn_state.pt` (the actual PyTorch
  checkpoint) is **not in git** — it's gitignored repo-wide (`*.pt`) and only exists on this
  filesystem at `data/cross_benzoin/cross_round9/gnn_attentive_9rounds_v1/models/gnn_state.pt`.
  If you're on a different machine/filesystem, you'll need to retrain the GNN (Section 4, step
  9) rather than expecting to load this file from git.
- `cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet` — the exact table used
  for both the tabular and GNN retrain (56,136 rows × 273 cols: 260 champion features + 13
  meta columns). This is **not in git** either (blanket `*.parquet` gitignore) — regenerate
  via Section 4 steps 7-8 if needed, or it should still exist on this filesystem.
- `data/cross_benzoin/cross_round8/...` — the round1-8 champion, same structure, for
  comparison.

To load the champion for prediction, use `cross_benzoin/predict_cross_champion.py`'s
`CrossBenzoinBlendPredictor.load(champion_dir, gnn_dir=...)` — it auto-detects the GNN
architecture (default vs attentive) from `gnn_norm_stats.joblib`.

---

## 3. Round10 decision evidence (learning curve)

`cross_benzoin/learning_curve_check_ensemble.py` on the round1-9 slim260 table, 5-fold ×
5-repeat group-CV at increasing data fractions (production MLP+XGB ensemble architecture):

| frac | n_pairs | n_rows | MAE | RMSE | R² |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 7,020 | 14,034 | 1.842 | 2.515 | 0.765 |
| 0.50 | 14,040 | 28,063 | 1.797 | 2.466 | 0.781 |
| 0.75 | 21,061 | 42,101 | **1.770** | **2.435** | **0.796** |
| 1.00 | 28,081 | 56,136 | 1.773 | 3.191 | 0.655 |

**Interpretation**:
- MAE improves monotonically from 0.25 → 0.75 (~4% relative gain, 1.842 → 1.770). Not
  plateaued through that range.
- At frac=1.0, MAE holds flat (1.773 vs 1.770) but R²/RMSE degrade sharply. This is **not
  new instability** — round6's own learning curve (`data/cross_benzoin/cross_round6/
  learning_curve_ensemble.csv`) shows the identical shape. The mechanism: at frac=1.0 there's
  no sampling freedom left — the "subsample" IS the full population, so it necessarily
  includes every rare heavy-tail case (e.g. phosphorus-containing outliers, the known worst
  error driver per `cross-five-diagnostics-20260717`), whereas random subsamples at lower
  fractions can luckily exclude them. Lower fractions' cleaner-looking numbers partly reflect
  this survivorship, not necessarily a truer signal.
- **Net read**: the trend through 0.75 is real and not yet flat, and full-data MAE doesn't
  degrade — modest but genuine evidence that a round10 (more data of the same kind) could
  still move the needle. This is NOT a strong "you must do round10" signal; it's evidence
  that the returns haven't visibly run out yet. Whether the marginal MAE gain (likely on the
  order of 0.01-0.03, extrapolating the 0.75→1.0 flattening trend) is worth another full
  DFT-SP campaign's cost is a judgment call for the user, not something the data alone
  resolves.

**This decision has not been made yet.** If the user says "go", proceed with Section 4.
If "no", the round1-9 champion above is production-final for now — no further action needed.

---

## 4. Round10 recipe (if greenlit) — reuses round9's exact scripts

Everything below is the SAME pipeline just run for round10 instead of round9. Script paths,
argument names, and gotchas are all confirmed working as of round9 (2026-07-21). Substitute
`9` → `10` and `8` → `9` in paths/args as needed.

### 4.0 — BEFORE drawing the round10 pool: apply the deferred cb_featurize fix

This session found (and deliberately did NOT apply mid-array, to avoid inconsistent behavior
within round9's already-running job) a real inode-footprint inefficiency in
`pipeline/compute/conf_funnel.py` and `conf_funnel_v2.py`: the `_ff`/`_spj`/`_opt` map
functions build per-conformer scratch directories (`ff{i:03d}`, `sp{i:03d}`) that are NOT
cleaned up until the ENTIRE molecule's conformer search + descriptors + ohess finishes. For a
flexible molecule (300-600 conformers), this was directly observed to leave 300+ conformer
subdirectories (1,000+ files) coexisting simultaneously for a SINGLE molecule, multiplied by
`workers=12` molecules per task and up to 8 co-located tasks per node — a large multiplier on
peak inode usage. **Apply this fix before round10's featurize array**: after each conformer's
energy is read in `_ff`/`_spj`, `shutil.rmtree()` its own subdirectory immediately (keep only
the final `opt{i:02d}` dirs for the L10 survivors until the molecule finishes). This is a pure
efficiency fix, doesn't change any computed values.

### 4.1 — Draw the round10 pool

Adapt `cross_benzoin/sample_round9_from_candidates_v3.py` (bump filename/seed, update
`--exclude-pairs` to include BOTH round8's AND round9's already-drawn pairs so round10 doesn't
re-draw them). Decide the target size (round9 was 8,000 pairs / 16,000 directed rows,
category-balanced + phosphorus-stratified) — whether to keep the same size or adjust is a
separate decision from the go/no-go itself.

### 4.2 — Featurize array

`cross_benzoin/slurm/submit_cb_featurize_array.sh`. **MANDATORY**: run `sinfo -p genoa -o "%20N
%10D %6t %C"` first, size the array count off the actual pool, and use `%60` throttle
regardless of idle node count (this constraint is `/scratch-local` per-node inode quota, not
cluster capacity — confirmed twice this session already). CHUNK=30 (not the old default 100 —
already the script's current default, changed 2026-07-20 for tail-latency reasons). Before
submitting with `REQUIRE_CACHE_COMPLETE=1`, run
`cross_benzoin/check_aldehyde_cache_coverage.py --pairs <round10 pairs.csv> --cache <ALD_CACHE>`
first (a small ~0.15% aldehyde-cache coverage gap exists and will hard-fail whole chunks
otherwise).

If a retry is needed (disk-quota-exceeded errors on `/scratch-local`, error shape
`exception:[Errno 122] Disk quota exceeded: ...`): identify failed pairs by matching
`(donor_smiles, acceptor_smiles)` against the retry INPUT list you generate — do NOT trust the
`id` column on error rows (whole-row-crash errors get a bare `{"id": <positional-index>,
"error": ...}` dict with no donor/acceptor info, from `cb_featurize.py`'s outer exception
handler — this is exactly what happened in round9). Use
`cross_benzoin/merge_round9_products.py` as the template for a `merge_round10_products.py`
that keys the merge off the retry-pairs INPUT file, not off detecting errors in the output.

### 4.3 — Product BDE array

`pipeline/slurm/submit_bde_gxtb_product_cross.sh`, CHUNK=50, genoa, cheap (1 CPU/task, ~1hr
walltime). No merge step needed — `assemble_cross_round_features.py` and
`assemble_cross_training_table_v3.py` both glob `bde_gxtb/chunk_*.csv` directly.

### 4.4 — Product mordred array

`cross_benzoin/slurm/submit_mordred_cross_products.sh`, CHUNK=100, genoa, very cheap
(~1s/molecule). After it completes, concat the chunks into a single sidecar CSV (simple
`pd.concat` over `mordred_products/chunk_*.csv`, write to
`data/cross_benzoin/cross_round10/round10_products_mordred.csv` — see round9's inline concat
for the exact 3-line pattern, no dedicated script exists, just write it inline).

### 4.5 — Assemble round10 CANDIDATE features (pre-DFT, for active-learning scoring)

`cross_benzoin/assemble_cross_round_features.py --round 10 --products-csv
<round10 merged products csv> --product-mordred-csv <round10 mordred sidecar>`. **Do NOT
confuse this with `assemble_cross_training_table_v3.py`** — that one is the POST-DFT final
table assembler and requires a `{rtag}_dft_products.csv` that doesn't exist yet at this stage.
Copy the output to the default-convention filename `cross_round10_features.parquet` (no
`--products-csv`-derived suffix) so `score_round_active_learning.py --round 10` finds it
without an explicit `--candidates-path` override.

### 4.6 — Active-learning score the round10 pool

`cross_benzoin/slurm/submit_score_active_learning.sh` with `sbatch --export=ALL,TAG=
cross_round10,TRAIN_TABLE=<round1-9 slim260 table path>,FEATURE_LIST=data/cross_benzoin/
cross_round9/scaffold_disjoint_9rounds_v1/models/feature_list.json,MODEL=ensemble,N_BOOT=40,
N_SELECT=<pool size>,SEED=42 cross_benzoin/slurm/submit_score_active_learning.sh`.
**CRITICAL**: use `--export=ALL,VAR=val,...` explicitly on the `sbatch` command line — inline
`VAR=val sbatch script.sh` does NOT forward environment variables to the SLURM job (this
failed instantly on round9's first attempt, caught by the script's own `${VAR:?...}` guard).
Score against the round1-9 champion (not round1-8's, since round1-9 is now the better
reference model).

### 4.7 — DFT-SP array

`pipeline/slurm/submit_dft_sp_cross_array.sh`, CHUNK=100, genoa, 48 CPU/task, 6hr walltime.
Pass the FULL merged products CSV as `--products-csv` (not a filtered subset) —
`dft_sp_cross_from_geom.py`'s `build_manifest()` self-filters error/missing-xyz rows
internally, so passing the unfiltered file is correct and simpler. This is real ORCA
(r2SCAN-3c/CPCM-DMSO) compute — budget hours, not minutes, per chunk.

Once complete: `pd.concat` the `chunk_*.csv` files into
`data/raw/dft_sp_cross/cross_round10/cross_round10_dft_sp.csv` (matches
`assemble_cross_training_table_v3.py`'s `round_paths()` expected naming), and copy the merged
products CSV to `data/cross_benzoin/cross_round10/cross_round10_dft_products.csv` (confirmed
this is literally just an alias copy of the products file — `load_round()` does its own
DFT-label inner-join internally against the separate `dft_csv` argument, no extra join step
needed).

### 4.8 — Assemble the round1-10 table

`cross_benzoin/assemble_cross_training_table_v3.py --rounds 1 2 3 4 5 6 7 8 9 10
--product-mordred-csv data/cross_benzoin/cross_round3/rounds123_products_mordred.csv
data/cross_benzoin/cross_round4/round4_products_mordred.csv
data/cross_benzoin/screen10k/screen10k_products_mordred.csv
data/cross_benzoin/cross_round8/round8_products_mordred.csv
data/cross_benzoin/cross_round9/round9_products_mordred.csv
data/cross_benzoin/cross_round10/round10_products_mordred.csv`

### 4.9 — Relabel + prune to the frozen 260-feature schema

Copy `cross_benzoin/relabel_scaffold_split_9rounds.py` → `relabel_scaffold_split_10rounds.py`,
bump the round9→round10 paths. Then prune to the SAME 260 features from
`data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json` (the
frozen champion schema, carried forward unchanged since round7) plus the 13 meta columns
(`id, donor_id, acceptor_id, pair_key, reaction_type, round, donor_smiles, acceptor_smiles,
smiles, dG_xtb_kcal, dG_gxtb_kcal, dG_orca_kcal, new_scaffold_split`). **This pruning step is
mandatory, not optional** — `train_scaffold_disjoint.py`'s `_feature_blocks()` dynamically
resolves feature columns from whatever table it's given, and an unpruned N-round table can
reintroduce an all-NaN-in-clean-train mordred column that crashes `MLPRegressor` (this
happened at round8, root-caused and fixed by pruning — see
`round8-complete-attentive-champion-and-round9-inflight-20260720.md` memory for the full
story). Sanity-check after pruning: `train[feats].isna().all()` should be empty for every
champion feature on the clean-train rows.

### 4.10 — Retrain champion + ensemble

`cross_benzoin/train_scaffold_disjoint.py --table <round1-10 slim260 table> --outdir
data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_v1`

### 4.11 — Retrain the attentive GNN

Copy `cross_benzoin/slurm/submit_gnn_attentive_9rounds.sh` → `_10rounds.sh`, bump paths
(gpu_a100 partition, 1 GPU, 2hr walltime — check `sinfo -p gpu_a100` first). Same
hyperparameters: `--arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed 0`.

### 4.12 — Bootstrap-verify the blend

Copy `cross_benzoin/verify_and_bootstrap_9rounds.py` → `_10rounds.py`, bump round9→round10
paths. B=20000, same methodology. Report the new MAE and P(blend better than ensemble-only).

### 4.13 — Learning-curve check again

`cross_benzoin/learning_curve_check_ensemble.py` on the round1-10 slim260 table (same SLURM
wrapper as Section 3), to inform the round11 decision the same evidence-based way.

### 4.14 — Commit + push

Follow the same grouped-commit pattern as this session (raw data landing, retrain+champion,
learning-curve — 3-4 logical commits, review `git status`/`git diff` before staging, don't
`git add -A` blindly). Check `.gitignore` already covers: nested `chunk_*/` dirs at any depth,
`*.parquet`, `*.pt`, large regenerable feature-join CSVs (add a new
`/data/cross_benzoin/cross_round10/cross_round10_features_*.csv` line if an analogous
140MB-class intermediate appears again).

---

## 5. Standing operational rules (apply to ALL of the above)

- **Before every `sbatch` with a concurrency/throttle/array-size parameter**: run `sinfo -p
  <partition> -o "%20N %10D %6t %C"` (and `squeue -u $USER -p <partition>` if useful) FIRST.
  Never copy a throttle number from a docstring or a "what worked last round" habit. `genoa`
  QoS has `MaxJobsPU=128` (confirmed via `sacctmgr`); `fat_genoa`/`fat_rome` have NO per-user
  job-count cap (confirmed live with 219 concurrent jobs) but are much smaller node pools
  (48/72 nodes vs genoa's 300+ idle) — still check `sinfo` there.
- **`cb_featurize` arrays specifically**: always `%60` throttle regardless of idle node count —
  this is a per-node `/scratch-local` inode-quota constraint, not cluster capacity.
- **`sbatch --export=ALL,VAR=val,...`** must be explicit whenever a submit script reads a
  required env var via `${VAR:?...}` — inline `VAR=val sbatch script.sh` does NOT forward it.
- **Never cancel a near-complete job** to fix a suboptimal parameter — let it finish, fix the
  next submission.
- **Preserve output history** — new filenames for reruns, never overwrite/delete prior
  results.

---

## 6. Where to find more detail

- `/home/schen3/.claude/projects/-gpfs-scratch1-shared-schen3/memory/round9-complete-round10-evidence-20260721.md`
  — the full memory entry this handoff is derived from (more narrative detail on each finding).
- `/home/schen3/.claude/projects/-gpfs-scratch1-shared-schen3/memory/round8-complete-attentive-champion-and-round9-inflight-20260720.md`
  — round8's own landing + the architecture-sweep evidence for why attentive-pooling is used.
- `cross_benzoin/docs/STATUS_EN.md` / `STATUS_ZH.md` — running project status (may lag this
  handoff by the time you read it; this handoff is more current as of 2026-07-21).
