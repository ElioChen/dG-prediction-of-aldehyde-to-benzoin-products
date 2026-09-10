# homo-only ΔG model (standalone)

Predict homo-benzoin ΔG on **homo data alone**, kept fully separate from the
cross pipeline. Same 260-feature schema + scaffold-disjoint split as the cross
champion so the holdout numbers are directly comparable.

## Data scope (2026-09-10)

Only the **30,000-row surviving-label subsample** (`homo_unify/homo_unify_v1_*`)
is trainable — the full ~219k homo library lost its DFT ΔG labels in the 2026-07
purge (physically gone, not in any backup; recompute is the approved post-Tier-B
campaign). All *features* (aldehyde/product QM, Mordred, BDE, geometry) are
present at full-library scale — see `homo_v6/LIBRARY_MANIFEST.md`.

- split (from `homo_unify_v1_scaffold_split_lookup.csv`): train 21,967 / test 4,038 / validation 3,995
- category-balanced: aliph-aliph / carbo-carbo / hetero-hetero, 10,000 each
- label `dG_orca_kcal` (r2SCAN-3c/CPCM-DMSO): median 4.28, std 6.84 kcal/mol, heavy tails
- baseline for Δ-learning: `dG_gxtb_kcal`

## Pipeline

| step | script | job | output |
|---|---|---|---|
| assemble table | `cross_benzoin/assemble_homo_standalone_table.py` | 26537710 (fat_genoa) | `homo_standalone_train_table_slim260.parquet` |
| tabular: single-XGB + MLP+XGB ensemble (Δ-learning) | `cross_benzoin/train_scaffold_disjoint.py` | 26537710 | `tabular/models/` + `tabular/models/metadata.json` |
| GNN: attentive-pooling TripleGNN + blend check | `cross_benzoin/train_cross_gnn_arch_sweep.py` | 26537713 (fat_rome, dep afterok) | `gnn_attentive_seed0/` |

Submit scripts: `cross_benzoin/slurm/submit_homo_standalone.sh` (+`_gnn.sh`).

## Results

_pending jobs 26537710 / 26537713._

Compare against: cross champion blend MAE **2.215** (n=448), g-xTB baseline
MAE 5.037. Prior full-library homo dG model (pre-purge) sat around MAE ~10 on a
wider distribution (memory `homo-active-relabel-null-result`).
