#!/bin/bash
#SBATCH --job-name=bde_gnn_seeds
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/bde_gnn_seed_%A_%a.out
#SBATCH --array=1-2
#
# Repeated-seed robustness check for B4 (2026-07-15): seed=0 already gave R2=0.775
# (aldehydes). Per this project's own established practice (delta_core.cv_evaluate's
# RepeatedKFold, explicitly to "damp split-noise that makes a single split MAE
# unreliable"), a single cold-start split isn't enough to trust a headline number --
# this runs 2 more seeds (array idx 1,2 -> --seed 1,2) so the three together give a
# mean +/- spread instead of one number that could just be split-luck.
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
BDE="$REPO/pipeline/bde"
mkdir -p "$REPO/runs/logs"

SEED="$SLURM_ARRAY_TASK_ID"
WHICH="${WHICH:-aldehydes}"

echo "D-MPNN BDE seed robustness ($WHICH, seed=$SEED) $(date)"
$ENV/bin/python -u "$BDE/train_gnn_bde.py" \
    --which "$WHICH" --seed "$SEED" \
    --out "$REPO/runs/logs/bde_gnn_${WHICH}_seed${SEED}_result.json"
echo "seed $SEED done $(date) exit=$?"
