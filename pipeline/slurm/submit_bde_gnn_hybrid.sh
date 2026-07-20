#!/bin/bash
#SBATCH --job-name=bde_gnn_hybrid
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/bde_gnn_hybrid_%A_%a.out
#SBATCH --array=0-1
#
# B6 (pipeline/bde/train_gnn_hybrid_bde.py, 2026-07-15): D-MPNN graph encoder + H-SPOC
# local-3D descriptors fused via chemprop's x_d mechanism, same pattern as the production
# dG model's proven GNN+tabular hybrid (pipeline/gnn/gnn_core.py). aldehyde formyl C-H
# (array idx 0), product ketC-carbC (array idx 1).
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
BDE="$REPO/pipeline/bde"
mkdir -p "$REPO/runs/logs"

WHICH=(aldehydes products)
W="${WHICH[$SLURM_ARRAY_TASK_ID]}"

echo "B6 hybrid D-MPNN+local3D BDE baseline ($W) $(date)"
$ENV/bin/python -u "$BDE/train_gnn_hybrid_bde.py" \
    --which "$W" --out "$REPO/runs/logs/bde_gnn_hybrid_${W}_result.json" \
    --pred-out "$REPO/runs/logs/bde_gnn_hybrid_${W}_test_predictions.csv"
echo "$W done $(date) exit=$?"
