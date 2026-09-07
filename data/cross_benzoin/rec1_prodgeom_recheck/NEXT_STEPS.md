# Rec-1 (B97-3c cheap-baseline lever) — decision tree after the recheck

Autonomous-session working doc (2026-09-08 overnight). Recheck = job **26454510**
(128 pilot holdout pairs, production funnel_v3 + GFN2-ohess geometry, r2SCAN-3c /
B97-3c / g-xTB SP on that geometry). Merge: `cross_benzoin/merge_rec1_prodgeom_recheck.py`.

## Read the verdict from `rec1_prodgeom_recheck_summary.json`

Key numbers: `prod_geom.overall.std_ratio_b973c_over_gxtb`, `.resid_b973c.std`,
`.resid_gxtb.std`, `.repro_r2scan.std` (funnel_v3 label-reproduction fidelity /
conformer noise — if this is large the residuals are contaminated by geom-regen noise).
Pilot reference on the SAME 128 pairs (one-shot geom): resid_gxtb 4.32 / resid_b973c
1.11 / ratio 0.26.

### GREEN  (ratio ≤ 0.45, resid_b973c std ≤ 2.5, repro_r2scan std ≤ 3.5)

B97-3c keeps its low scatter on the real production geometry → the lever is real.
**Do NOT immediately launch the full 35k recompute** (multi-day, ~100k+ SP,
cluster-week). Instead the decisive intermediate:

1. **B97-3c dG for the frozen scaffold-disjoint holdout (n=448) + a train slice**
   big enough to retrain the Δ-model. Reuse `rec1_prodgeom_recheck_worker.py`
   (it already produces `dG_b973c_kcal` per pair on production geometry). Pair list:
   the 448 holdout + ~2-4k scaffold-disjoint train rows from
   `cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet`
   (`new_scaffold_split`), preferring rows whose product geom is archived (see
   coverage below) so most tasks are SP-only, regen only the gaps.
2. **Retrain the Δ-model with `dG_pred = dG_b973c + ML correction`** (swap the g-xTB
   baseline column), scaffold-disjoint, and compare holdout MAE to champion 2.215.
   Scripts: `cross_benzoin/finalize_correction.py` / the ensemble trainer used for
   `ensemble_scaffold_disjoint.joblib`.
3. Drop to ~1.5 or below → project-level result → THEN scope the full recompute for a
   human go. Sub-1-bootstrap-SE change → the std-ratio was a mirage at the dG level
   (constant-offset absorption + conformer floor), write it up, move to Rec 2.

Geometry for the intermediate: **must be regenerated** — see the coverage finding below.

### ⚠ Geometry-availability finding (2026-09-08, `rec1_fullset_geom_coverage.py`)

Archived funnel_v3 geometry coverage for the 35,528-row training table is
**effectively zero**:

| | product geom | donor ald | acceptor ald | **all 3** |
|---|--:|--:|--:|--:|
| full table | 35.8% | 5.6% | 5.6% | **0.6%** (210) |
| test holdout (n=448) | — | — | — | **0%** |

The funnel_v3 geometries were purged in 2026-07 and are not recoverable (only r8/r9
have partial aldehyde coverage; the round tarballs hold products for ~36% but the
aldehyde side is gone). **Consequence: EVERY B97-3c dG — intermediate or full — needs
funnel_v3 (+ohess or +opt) geometry regeneration from SMILES**, i.e. the recheck
worker's exact path, at ~1-2 h/pair.

Re-scoped cost:
- **Intermediate** (448 holdout + ~1-2k train, all regen): ~1500-2500 pair-tasks
  × ~1.5 h ≈ 1-2 cluster-days at rome %150.
- **Full 35,528-pair recompute, all regen**: a **multi-week** cluster campaign
  (the handoff's "~1 week" assumed archived geoms + SP-only, which do not exist).
  Needs a human go with this cost on the table. Optimisation levers: trust the
  **stored xTB RRHO thermal** (in the training table `*_G_xtb` cols /
  `aldehydes_all.csv`) and skip ohess → funnel_v3 rank-0 (GFN2 --opt) geom + B97-3c
  SP only, ~45-75 min/pair; shard wide; cpus-per-task 8.

### AMBER  (0.45 < ratio ≤ 0.7)

B97-3c still helps but less than the pilot's 0.26. Same intermediate as GREEN step 1-2
but be stricter: only worth the full recompute if the retrained Δ-model holdout MAE
actually beats 2.215 by > 1 bootstrap SE (~0.15). Otherwise document and move to Rec 2.

### RED  (ratio > 0.7, or repro_r2scan std ≫ 3.5 making it uninterpretable)

The pilot's 1.11 was a conformer artefact of the shared one-shot geometry; on the real
production geometry B97-3c ≈ g-xTB as a baseline. **Close the B97-3c direction.**
Update `data/analysis/homo_cross_gap/FINDING.md` §Task D and `PROJECT_SUMMARY` §6/§9.
Pivot the session to **Rec 2** (homo_unify 30k → 260-schema featurize → down-weighted
merge → champion retrain; the user deferred launching this pending Rec 1, so on RED it
becomes the main line — but confirm with the user before submitting the large featurize
array, per the earlier decision).

## Full B97-3c recompute (only after a human go on GREEN + intermediate win)

- Scope: 35,528 training pairs × 3 species, B97-3c/CPCM(DMSO) SP on production
  funnel_v3+ohess geom. Reuse stored r2SCAN label if repro_r2scan std small; else
  co-recompute r2SCAN for a self-consistent relabel.
- Strategy S2 (archived-geom SP-only) vs S1 (regen) vs S3 (hybrid) decided by
  `fullset_geom_coverage.txt`.
- Add `dG_b973c_kcal` to `assemble_cross_training_table_*` / champion feature pipeline.
- Combine with Rec-3 direction 1 (targeted baseline_risk heteroatom substructure
  enrichment) in the same array batch.
- **Trap**: any change to `homo_v6/aldehydes_all.csv` / `products_all.csv` schema →
  run the `predict_dg.py` 20-pair smoke test (memory `predict-dg-g-gxtb-regression-fixed`).
