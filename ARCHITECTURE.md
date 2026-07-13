# benzoin_dG — architecture & data/script map

Predict benzoin-condensation **ΔG** from an aldehyde SMILES: cheap xTB ΔG + a
**Δ-learning** correction to DFT. **Canonical DFT level = r2SCAN-3c** (composite,
CPCM/DMSO, conformer-ensemble `k=3`), computed by the **unified featurizer**.

## Canonical pipeline (use this path)

```
filter_smiles.py            218k aldehyde pool            data/library/aldehydes_clean.csv
   │  expand_subset.py (MaxMin diversity, incremental)
   ▼
data/library/subset_v*.csv  representative subset (the "budget" of molecules to label)
   │  slurm/run_featurize_array.sh  → compute/featurize.py   (r2SCAN-3c, ensemble_k=3)
   │     ONE shared xTB geometry → 63 descriptors + dG_xtb + dG_orca per molecule
   ▼
data/raw/featurize_*/mol_*/features.csv   per-molecule unified output
   │  (assemble → parquet)        ⚠ no dedicated script yet — see "Gaps"
   ▼
data/featurize_v2.parquet   THE training table (descriptors + labels, geometry-consistent)
   │  delta_core.load_training_table → cv_evaluate
   ▼
trees:  train_delta.py / sweep_delta.py        GNN: gnn/train_gnn.py / sweep_gnn.py
   │  assemble_model.py → ship winner to src/benzoin_dG/models/
   ▼
src/benzoin_dG/  (installable inference package)
```

## DATA — classified

### ✅ Canonical (r2SCAN-3c, unified featurize)
| path | what | n |
|---|---|---|
| `data/library/aldehydes_clean.csv` | filtered aldehyde pool | 218k |
| `data/library/subset_v2.csv` | current labeled subset | 500 |
| `data/library/subset_expansion_v3.csv` / `subset_v4.csv` | +1000 new / 1500 combined | 1000 / 1500 |
| `data/raw/featurize_full/mol_*/features.csv` | per-mol r2SCAN-3c output (the 500) | 500 |
| `data/raw/featurize_v3/mol_*/features.csv` | per-mol r2SCAN-3c output (the +1000, **running**) | →1000 |
| **`data/featurize_v2.parquet`** | **assembled training table** (after QC) | 474 |
| `data/labels/chunk_000`, `data/descriptors/chunk_000` | what `delta_core` reads today — an **exact extract of featurize_v2** (verified 474/474); redundant once the loader reads the parquet | 474 |
| `data/labels/chunk_000/delta_G.r2scan3c_backup.csv` | safety backup of the r2SCAN-3c labels | — |

### ❌ Legacy / deprecated (old PBE0-D4 or pre-unified separate path)
| path | what | action |
|---|---|---|
| `data/raw/labels_ensemble/` | **58 GB** PBE0-D4 ensemble ORCA *work files* | **delete** |
| `data/raw/labels/` | PBE0-D4 ΔG raw (separate `thermo_orca` path) | archive/delete |
| `data/raw/descriptors/` | descriptors from the separate `ald_descriptors` path | archive/delete |
| `data/descriptors_pbe0_old/`, `data/labels_pbe0_old/` | explicitly-old PBE0 chunks | delete |
| `data/labels_partial/`, `data/chunks/` | early partial/staging | delete |
| `data/flexible/`, `data/raw/featurize_flex/` | flexible-geometry side experiment | delete (7 mols) |
| `data/raw/featurize/` | empty (logs only) | delete |

*The PBE0-D4 set is a different DFT level — never mix it into the r2SCAN-3c table.*

## SCRIPTS — classified

| role | current (use) | legacy (don't extend) |
|---|---|---|
| **selection** | `expand_subset.py` (MaxMin, incremental) | `select_subset.py` (PCA+KMeans) |
| **compute** | `compute/featurize.py` (unified: descriptors+ΔG, r2SCAN-3c) | `compute/ald_descriptors.py` (desc only), `compute/thermo_orca.py` (ΔG only — still imported *by* featurize) |
| **assembly** | *(needed: featurize_* → parquet)* | `merge_labels.py`, `add_expand_chunk.py` (chunk-CSV path) |
| **training** | `delta_core.py`, `train_delta.py`, `sweep_delta.py` (trees); `gnn/{gnn_core,train_gnn,sweep_gnn}.py` | — |
| **ship** | `assemble_model.py` → `src/benzoin_dG/models/` | — |
| **SLURM** | `slurm/run_featurize_array.sh` + `submit_featurize_array.sh`; `submit_gnn.sh` | `run_/submit_{descriptors,labels}_array.sh` (separate path), `submit_train.sh` |

## Cleanup status
1. ✅ **`pipeline/assemble_featurize.py`** added: `data/raw/featurize_*/mol_*/features.csv`
   → `data/featurize.parquet` (the single training table; dedup on `index`, QC stays in delta_core).
2. ✅ **`delta_core` prefers the parquet** (`DEFAULT_FEATURIZE_PARQUET`), chunk-CSV globs are
   the fallback. Shared QC/feature-build in `_finalize_table`. All trainers auto-use it.
3. ✅ **Legacy scripts quarantined** under `pipeline/legacy/` (`select_subset`, `merge_labels`,
   `add_expand_chunk`, `run_/submit_{descriptors,labels}_array.sh`) — see its README.
4. ⏸️ **Legacy/PBE0-D4 data dirs kept for now** (deferred). To free ~58 GB later, delete
   `data/raw/labels_ensemble/` + the `*_pbe0_old` / `labels_partial` / `chunks` / `flexible` dirs.
   (`ald_descriptors.py` + `thermo_orca.py` stay — they are imported by `featurize.py`.)

## Data cleaning v2 (2026-06-13, after the n=1500 energy diagnostic)
`pipeline/analyze_energies.py` showed the xTB→DFT correction is a near-constant
+11.5 kcal offset + ~7 kcal structure-independent noise, and that outliers cluster
in exotic groups (nitroso σ≈23, nitro σ≈15, azo σ≈13). So `filter_smiles.py` gained
3 rejection rules — **isotope, zwitterion_or_nitro, reactive_group** (nitroso/azo/
SF5/ketene/isocyanide). Re-clean: pool 217,975→**206,966** (`aldehydes_clean_v2.csv`);
labeled 1487→**1351** (delta_core auto-drops these via `filter_smiles.classify`,
`drop_reactive=True`). Cleaning improved xgb MAE 3.245→3.19. **Trees stay the
production model** — model zoo (15 families) shows ~3.2 MAE is a noise floor (even
ridge ties); GNN still +0.19 behind. Diagnostics: `runs/figs/{energy_analysis,
model_zoo,learning_curve}.png`.

## PRODUCTION MODEL (2026-06-13) — AROMATIC-ONLY, n=1627
Shipped `src/benzoin_dG/models/`: xgb Δ-model on **1627 aromatic aldehydes**
(1007 carbo + 620 hetero), **CV MAE 2.68** (R² 0.66, xTB base 12.6). Per-category
AD (`category_ad.json`): carbo ±2.50, hetero ±2.98. Inference rejects non-aromatic
SMILES (`scope.py`, `benzoin_relevant=False`).

**Why 2.68 << the old 3.21:** the aromatic subset was data-starved (n=604 → MAE 3.08);
tripling it via targeted expansion (813 carbo + 250 hetero) dropped MAE to 2.68 — the
aromatic learning curve was still steep (unlike the flat global noise floor). Full
parquet now 3063 molecules; training auto-filters to aromatic via `delta_core.CHO_SCOPE`.
Data assembled from `featurize_{full,v3,v4,aromatic_v1,hetero_v1}`. See
`pipeline/docs/findings_2026-06-13_aromatic_pivot.md`.

## Current jobs
- `23766389` — conformer-sensitivity probe: 50 aromatic re-featurized at N_CONFS=30,
  ENSEMBLE_K=10 (vs prod 10/3) → `data/raw/featurize_conftest/`. Tests whether the
  ~7 kcal residual is conformer-driven (raise k globally) or the xTB↔DFT method gap.

SLURM now defaults to **genoa** (192 cores/node, array `%192`); GNN stays gpu_a100.
