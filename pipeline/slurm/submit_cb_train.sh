#!/bin/bash
#SBATCH --job-name=cb_train
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/cb_train_%j.out
#
# Δ-learning sweep for the CROSS-BENZOIN product-descriptor era (g-xTB baseline).
# Sweep ONLY (no assemble_model -> does NOT overwrite the shipped production model).
# Writes the winner to a DEDICATED outdir so runs/models/ is untouched too.
#
#   sbatch pipeline/slurm/submit_cb_train.sh                  # g-xTB table (default)
#   sbatch --export=ALL,PARQUET=...,OUTDIR=...,TRIALS=120 pipeline/slurm/submit_cb_train.sh
set -euo pipefail
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
PARQUET="${PARQUET:-$REPO/data/featurize_cb_homo_train_gxtb.parquet}"
OUTDIR="${OUTDIR:-$REPO/runs_cb_gxtb}"
TRIALS="${TRIALS:-120}"
TARGET="${TARGET:-dG_orca_kcal}"
export MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-benzoin_cb_gxtb_product}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$REPO/runs/logs" "$OUTDIR/models"
cd "$REPO"
echo "=== cb Δ-train  job=${SLURM_JOB_ID:-local}  $(date) ==="
echo "parquet=$PARQUET  outdir=$OUTDIR  trials=$TRIALS  exp=$MLFLOW_EXPERIMENT"
"$PY" pipeline/sweep_delta.py --parquet "$PARQUET" --target "$TARGET" \
    --model all --trials "$TRIALS" --outdir "$OUTDIR"
echo "=== done $(date) — model in $OUTDIR/models/ (NOT shipped to src/) ==="
