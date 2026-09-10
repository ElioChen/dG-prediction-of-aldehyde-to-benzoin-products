# homo-only ΔG from scratch — full-library relabel + simple models

**User directive (2026-09-10):** redo homo from scratch on the *full* library;
recompute the DFT labels that were lost in the purge; use a **single XGBoost**
and a **single GNN**, reported independently — not the iterated champion blend.
**Launch: after cross Tier B drains** (user pick; ~09-11).

## Phase 1 — relabel campaign (the "补算")

The full-library homo DFT ΔG labels are physically gone (30k of ~219k survive).
Recompute, self-consistently, with the same protocol as cross Tier B.

| | |
|---|---|
| worker | `cross_benzoin/rec_homo_relabel_worker.py` — homo fork of the Tier B worker: **2 species/pair** {product, aldehyde} (not 3), `dG_lvl = (G_lvl[prod] − 2·G_lvl[ald])·627.509` |
| per species | conf_funnel_v3 rank → GFN2 `--ohess tight --alpb dmso` → r2SCAN-3c + B97-3c + g-xTB SP on that geometry (`_species` reused verbatim from the Tier B worker) |
| pair list | `cross_benzoin/build_homo_relabel_pairs.py` → `homo_relabel_pairs.csv` = **161,630 rows** (2,000 front-loaded QC pairs that already have a 30k label + **159,630 to relabel**) |
| array | `cross_benzoin/slurm/submit_homo_relabel.sh` — CHUNK 12 → 13,470 tasks, `%250`, `--array`/`--partition` overridable for multi-arm (same split style as Tier B: rome QOS-128 arm + fat_genoa arm) |
| shards | `data/cross_benzoin/homo_standalone/relabel/shards/shard_%05d.csv`, `.done` markers, pid-skip resume |
| scale | ~323k species SP (2 × 161.6k), ≈ Tier B × 4.5; **~3 weeks** at ~250 concurrent |
| split | `homo_v6/products_scaffold_split.csv` (train 175,636 / val 21,899 / test 21,886); ~1,086 relabel pairs have no split → impute/exclude |
| smoke | job 26538271 (genoa, 2 pairs) — verify before the full array |

**True failure** = col6 `dG_r2scan_kcal` empty (same convention as Tier B).
**QC:** the 2,000 front pairs carry `label` (surviving 30k value) → `repro_r2scan`
column = new − old; a tight distribution confirms the campaign reproduces the
old labels self-consistently before trusting the 159k new ones.

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
