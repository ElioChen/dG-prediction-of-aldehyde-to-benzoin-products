#!/bin/bash
#SBATCH --job-name=cross_gnn_wldn
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/cross_gnn_wldn_%A.out
#
# WLDN-style (shared-weight encoder + reaction-difference readout) comparison
# point for the cross-benzoin dG GNN leg, user-requested 2026-09-17 ("尝试其他
# 的基于反应的GNN"). See train_cross_gnn_wldn.py docstring for how this differs
# from both the champion TripleGNN (concat, separate weights) and CRG/CGR-delta
# (graph merge, one encoder).
#
# Env: /home/schen3/venv/nequip -- same env the champion TripleGNN and CRG/
# CGR-delta all train in, so this is an architecture-only comparison.
set -euo pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="/home/schen3/venv/nequip/bin/python"
CB="$REPO/cross_benzoin"
mkdir -p "$REPO/slurm_logs"

export CB_TARGET_COL=dG_r2scan_kcal
export CB_BASELINE_COL=dG_b973c_kcal

SEED="${SEED:-0}"
TAG="${TAG:-}"
OUT="$REPO/data/cross_benzoin/cross_round10/gnn_wldn_10rounds_b973c_seed${SEED}${TAG:+_$TAG}"

echo "cross-benzoin WLDN-GNN seed=$SEED $(date)"
$PY -u "$CB/train_cross_gnn_wldn.py" \
    --table "$REPO/data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet" \
    --feature-list "$REPO/data/cross_benzoin/feature_list_257_no_nCHO_v2.json" \
    --outdir "$OUT" \
    --hidden 128 --layers 4 --batch-size 64 --max-epochs 150 --patience 15 \
    --lr 1e-3 --seed "$SEED"
echo "done $(date) exit=$?"
