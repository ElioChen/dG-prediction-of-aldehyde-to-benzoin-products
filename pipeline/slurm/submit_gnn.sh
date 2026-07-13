#!/bin/bash
#SBATCH --job-name=dmpnn_sweep
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/dmpnn_%j.out
#
# Hybrid D-MPNN (Chemprop) on one A100. GPU budget is tight, so the DEFAULT is a
# single cheap CV train (MODE=train); the 40-trial sweep must be opted into.
# IMPORTANT: a bare `MODE=… sbatch` does NOT reach the job on this cluster — pass
# overrides through sbatch's own export list:
#   sbatch pipeline/slurm/submit_gnn.sh                                   # 1 train (default)
#   sbatch --export=ALL,MODE=sweep,TRIALS=40 pipeline/slurm/submit_gnn.sh # full sweep
#   sbatch --export=ALL,MODE=train,PARAMS='{"depth":6,"ensemble":2}' pipeline/slurm/submit_gnn.sh
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/gnn"
GNN="$REPO/pipeline/gnn"
mkdir -p "$REPO/runs/logs"

MODE="${MODE:-train}"          # train (cheap, default) | sweep (opt-in, ~2h)
TRIALS="${TRIALS:-40}"
REPEATS="${REPEATS:-2}"
ENS="${ENS:-3}"
TARGET="${TARGET:-dG_orca_kcal}"
PARAMS="${PARAMS:-{}}"
PARQUET="${PARQUET:-$REPO/data/featurize.parquet}"   # label version to train on

export PYTHONUNBUFFERED=1
# chemprop/lightning + the single visible A100.
echo "node=$(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)  $(date)"
cd "$GNN"

echo "Parquet: $PARQUET"
if [[ "$MODE" == "train" ]]; then
    "$ENV/bin/python" train_gnn.py --target "$TARGET" --repeats "$REPEATS" \
        --params "$PARAMS" --parquet "$PARQUET"
else
    "$ENV/bin/python" sweep_gnn.py --target "$TARGET" --trials "$TRIALS" \
        --repeats "$REPEATS" --ensemble-max "$ENS" --parquet "$PARQUET"
fi
echo "Done $(date)"
