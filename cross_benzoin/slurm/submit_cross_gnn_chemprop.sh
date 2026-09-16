#!/bin/bash
#SBATCH --job-name=cross_gnn_chemprop
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/cross_gnn_chemprop_%A.out
#
# Chemprop (D-MPNN, MulticomponentMPNN over product+donor+acceptor) comparison
# point for the cross-benzoin dG GNN leg, user-requested 2026-09-16 ("try
# chemprop on the main project"). See train_cross_gnn_chemprop.py's docstring
# for the architecture + why the split uses new_scaffold_split directly
# (train_cross_delta.pair_split_labels() is currently broken post-candidates_v3
# retirement -- this script avoids that path entirely).
#
# Env: /home/schen3/venv/bde_gnn (torch 2.13+cu130, chemprop 2.2.0, lightning --
# survived/rebuilt post-2026-07-purge; verified working here, unlike
# /gpfs/scratch1/shared/schen3/envs/bde_gnn and envs/gnn which are both corrupted
# (empty bin/), see [[envs-gnn-corrupted]]).
set -euo pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="/home/schen3/venv/bde_gnn/bin/python"
CB="$REPO/cross_benzoin"
mkdir -p "$REPO/slurm_logs"

export CB_TARGET_COL=dG_r2scan_kcal
export CB_BASELINE_COL=dG_b973c_kcal

SEED="${SEED:-0}"
OUT="$REPO/data/cross_benzoin/cross_round10/gnn_chemprop_10rounds_b973c_seed${SEED}"

echo "cross-benzoin chemprop GNN seed=$SEED $(date)"
$PY -u "$CB/train_cross_gnn_chemprop.py" \
    --table "$REPO/data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet" \
    --feature-list "$REPO/data/cross_benzoin/feature_list_257_no_nCHO_v2.json" \
    --outdir "$OUT" \
    --depth 4 --hidden 300 --ffn-hidden 300 --ffn-layers 2 --dropout 0.0 \
    --batch-size 64 --max-epochs 150 --patience 15 --seed "$SEED"
echo "done $(date) exit=$?"
