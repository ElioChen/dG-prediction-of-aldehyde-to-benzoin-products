#!/bin/bash
#SBATCH --job-name=bde_gnn
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/bde_gnn_%A_%a.out
#SBATCH --array=0-1
#
# Full-library D-MPNN BDE baseline (Phase-1 B4, BDE_prediction.md section 六): aldehyde
# formyl C-H BDE (array idx 0) and product ketC-carbC BDE (array idx 1), predicted
# directly from the 2D graph -- see pipeline/bde/train_gnn_bde.py for why no explicit
# bond-marking is needed (always the same fixed bond position per molecule class).
#
# Runs in its OWN isolated env (envs/bde_gnn), NOT the existing envs/gnn -- that env's
# stdlib was found corrupted (empty lib/python3.12/encodings/) while this work was being
# set up, unrelated to this job; flagged separately, left untouched rather than repaired
# out from under whatever else may depend on it.
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
BDE="$REPO/pipeline/bde"
mkdir -p "$REPO/runs/logs"

WHICH=(aldehydes products)
W="${WHICH[$SLURM_ARRAY_TASK_ID]}"

echo "D-MPNN BDE baseline ($W) $(date)"
$ENV/bin/python -u "$BDE/train_gnn_bde.py" \
    --which "$W" --out "$REPO/runs/logs/bde_gnn_${W}_result.json" \
    --pred-out "$REPO/runs/logs/bde_gnn_${W}_test_predictions.csv"
echo "$W done $(date) exit=$?"
