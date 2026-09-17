#!/bin/bash
#SBATCH --job-name=cross_gnn_cgr_delta
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/cross_gnn_cgr_delta_%A.out
#
# CGR-delta (explicit bond-order-change) comparison point for the cross-benzoin
# dG GNN leg, user-requested 2026-09-17 ("尝试其他的基于反应的GNN") -- the
# architecture refinement train_cross_gnn_crg.py's own docstring/LAB_JOURNAL
# flagged but didn't build (the order-changed carbC-hydO edge). See
# train_cross_gnn_cgr_delta.py + crg_builder.py's build_crg_delta() docstrings.
#
# Env: /home/schen3/venv/nequip -- same env the champion TripleGNN and plain
# CRG both train in, so this is an architecture-only comparison, no env confound.
set -euo pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="/home/schen3/venv/nequip/bin/python"
CB="$REPO/cross_benzoin"
mkdir -p "$REPO/slurm_logs"

export CB_TARGET_COL=dG_r2scan_kcal
export CB_BASELINE_COL=dG_b973c_kcal

SEED="${SEED:-0}"
OUT="$REPO/data/cross_benzoin/cross_round10/gnn_cgr_delta_10rounds_b973c_seed${SEED}"

echo "cross-benzoin CGR-delta-GNN seed=$SEED $(date)"
$PY -u "$CB/train_cross_gnn_cgr_delta.py" \
    --table "$REPO/data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet" \
    --feature-list "$REPO/data/cross_benzoin/feature_list_257_no_nCHO_v2.json" \
    --outdir "$OUT" \
    --hidden 128 --layers 4 --batch-size 64 --max-epochs 150 --patience 15 \
    --lr 1e-3 --seed "$SEED"
echo "done $(date) exit=$?"
