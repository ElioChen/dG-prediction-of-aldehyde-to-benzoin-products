#!/bin/bash
#SBATCH --job-name=cross_gnn_crg
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/cross_gnn_crg_%A.out
#
# Condensed reaction graph (CRG) comparison point for the cross-benzoin dG GNN
# leg, user-requested 2026-09-16. See train_cross_gnn_crg.py + crg_builder.py
# docstrings for the architecture and the atom-economical-reaction argument for
# why the product's own graph already contains both donor+acceptor atoms intact.
#
# Env: /home/schen3/venv/nequip -- the SAME env the deployed champion TripleGNN
# trains in (torch 2.11 + PyTorch Geometric 2.7), so this is an architecture-only
# comparison, no env confound (unlike the chemprop comparison point, which
# necessarily uses a different env).
set -euo pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="/home/schen3/venv/nequip/bin/python"
CB="$REPO/cross_benzoin"
mkdir -p "$REPO/slurm_logs"

export CB_TARGET_COL=dG_r2scan_kcal
export CB_BASELINE_COL=dG_b973c_kcal

SEED="${SEED:-0}"
TAG="${TAG:-}"
OUT="$REPO/data/cross_benzoin/cross_round10/gnn_crg_10rounds_b973c_seed${SEED}${TAG:+_$TAG}"

echo "cross-benzoin CRG-GNN seed=$SEED $(date)"
$PY -u "$CB/train_cross_gnn_crg.py" \
    --table "$REPO/data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet" \
    --feature-list "$REPO/data/cross_benzoin/feature_list_257_no_nCHO_v2.json" \
    --outdir "$OUT" \
    --hidden 128 --layers 4 --batch-size 64 --max-epochs 150 --patience 15 \
    --lr 1e-3 --seed "$SEED"
echo "done $(date) exit=$?"
