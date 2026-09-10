# homo-only ΔG from scratch — full-library relabel + simple models

**User directive (2026-09-10):** redo homo from scratch on the *full* library;
recompute the DFT labels that were lost in the purge; use a **single XGBoost**
and a **single GNN**, reported independently — not the iterated champion blend.
**Launch: after cross Tier B drains** (user pick; ~09-11).

## Phase 1 — relabel campaign (the "补算") — **FAST SP-ONLY ROUTE** (adopted 2026-09-10)

The full-library homo DFT ΔG labels are physically gone (30k of ~219k survive,
no backup anywhere — git / scratch / home all checked). Recompute by **DFT
single-point on the archived 2026-09 GFN2-opt geometries**, reusing the stored
xTB RRHO thermal — NO conformer search, NO Hessian. This is how the original
homo labels were made (memo `full-dft-sp-funnelv3-autolaunch`): ~10-20× cheaper
than the self-consistent regen, and geometry-consistent with the g-xTB /
descriptor library.

| | |
|---|---|
| geometry store | `data/cross_benzoin/bde_homo_product_featurize_20260902/chunk_*/geom.tar.zst` (members `xyz/pNNNNNN.xyz` + `ald_xyz/aNNNNNN.xyz`); home copy `~/benzoin_backups/bde_homo_rebuild_20260902/homo_product_chunk_geoms_20260902.tar` |
| thermal (reused) | product `G_product − xtb_energy`, aldehyde `G_xtb − xtb_energy` (from the `*_all.csv`) |
| manifest | `cross_benzoin/build_homo_sp_manifest.py` → `relabel_sp/homo_sp_manifest.{parquet,csv}` = **184,052 rows** (24,554 carry a surviving 30k label = QC block, front-loaded; split train 145,940 / test 18,529 / val 18,498 / 1,085 nan) |
| worker | `cross_benzoin/homo_sp_from_geom_worker.py` — extract prod+ald xyz from the zst, `calc_orca_sp` r2SCAN-3c (label) + B97-3c (Δ-baseline) per species, `dG_lvl = (G_lvl[prod] − 2·G_lvl[ald])·627.509`; append+fsync, resume by `id` |
| array | `cross_benzoin/slurm/submit_homo_sp.sh` — CHUNK 20 → 9,203 tasks, `%250`, 24 cpu, `--sp-workers 16`; `--array`/`--partition` overridable for multi-arm |
| shards | `relabel_sp/shards/shard_%05d.csv`, `.done` markers |
| scale | ~184k pairs × 2 species × 2 methods ≈ 736k SPs, but SP-only → **~2-3 days** at ~250 concurrent (original 1-method 219k run was ~18h / ~100k core-h) |
| smoke | job 26539727 (genoa, 3 QC pairs) — verify before the full array |

**True failure** = `dG_r2scan_kcal` empty. **QC:** the 24,554 label-carrying rows
give `repro_r2scan = new − stored`; expect a possible constant offset vs the
old 30k (different geometry vintage) — a *tight* distribution around whatever
the offset is confirms the SPs are clean. Absolute offset is harmless for a
from-scratch model (learned target). **Do not merge new + old 30k labels.**

**Known geometry-archive gap (~5-10%, found 2026-09-10):** the 2026-09
`bde_homo_product_featurize` geom archive is partial for chunks ~1400-1830
(chunk_1650 has 13/100 product xyz, 1550 has 29, 1700 has 69) plus ~59 chunks
with no `geom.tar.zst`. Those pairs record `geom_extract_fail` and are dropped
by `merge_homo_sp.py` -- the model trains on ~165-175k of 184k, still a strong
full-library set. Top-up (post-campaign, if 100% wanted): GFN2-opt the
`geom_extract_fail` ids from SMILES (plain `xtb --opt`, NO Hessian, reuse the
stored xTB thermal), then re-run `homo_sp_from_geom_worker` on that sub-manifest.

### superseded: self-consistent regen route
`rec_homo_relabel_worker.py` + `build_homo_relabel_pairs.py` +
`submit_homo_relabel.sh` (conf funnel + ohess + 3 SP per species, ~3 weeks) —
kept in the repo but NOT the plan. Its 2-pair smoke (job 26538271) showed a
consistent ~−5.5 kcal offset vs the stored 30k labels (geometry-protocol
difference). Only revisit if cross-Tier-B-level label accuracy is wanted.

## Phase 2 — merge + assemble (build while Phase 1 runs)

- `merge_homo_relabel.py` (to write) — concat shards → dedupe by pid →
  `homo_relabel_labels.csv` (`pid, dG_r2scan_kcal, dG_b973c_kcal, dG_gxtb_kcal, resid_*, repro_*`)
  + coverage / QC-block repro summary.
- extend `assemble_homo_standalone_table.py` with `--labels <merged>` +
  `--full-library` → full ~160k table, champion 260-feature schema,
  `new_scaffold_split` from the full-library split, `dG_r2scan_kcal` target +
  `dG_b973c_kcal` Δ-baseline.

## Phase 3 — two simple models, reported independently (no blend)

| model | script | notes |
|---|---|---|
| single XGBoost Δ-model | `train_scaffold_disjoint.py` (its single-XGB head), `CB_TARGET_COL=dG_r2scan_kcal CB_BASELINE_COL=dG_b973c_kcal` | report the `scaffold_disjoint_holdout_xgb` MAE/R²; the MLP+XGB ensemble it also emits is context only |
| single attentive GNN | `train_cross_gnn_arch_sweep.py --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed 0` | report `gnn_only_mae`; 1 seed, no ensemble, blend line is informational |

Submit: `submit_homo_standalone.sh` (assemble+XGB, fat_genoa/rome) → dependent
`submit_homo_standalone_gnn.sh` (GNN, fat_rome). Both already written; repoint
`TABLE` at the full-library table and set the `CB_*` env once Phase 2 lands.

## Benchmarks

cross champion blend MAE 2.215 (n=448) · g-xTB baseline 5.037 · pre-purge
full-library homo model ≈ MAE 10 on a wider distribution.
Target: honest scaffold-disjoint holdout MAE for the two simple models on the
full self-consistently-relabelled homo library.
