#!/bin/bash
#SBATCH --job-name=bz_explore
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/explore_%j.out
#
# CPU model exploration for benzoin ΔG on the 1500-row r2SCAN-3c table. Three
# steps, all CPU (no GPU budget), submitted as ONE SLURM job:
#   [1] sweep_delta.py    — re-tune + ship the production xgb tree (the baseline)
#   [2] explore_models.py — regression zoo + uncertainty + classification framings
#   [3] learning_curve.py — CV MAE vs n (tree) with GNN points overlaid
#
#   sbatch pipeline/slurm/submit_explore.sh
#   sbatch --export=ALL,TRIALS=120 pipeline/slurm/submit_explore.sh
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/gnn"
PY="$ENV/bin/python"
TRIALS="${TRIALS:-80}"
TARGET="${TARGET:-dG_orca_kcal}"
export MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-benzoin_delta_dG_1500}"
export PYTHONUNBUFFERED=1
# Let xgboost/sklearn manage cores via n_jobs=-1; cap the BLAS thread pools so
# they don't oversubscribe the 64 allocated cores.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

mkdir -p "$REPO/runs/logs"
cd "$REPO/pipeline"
echo "=== bz_explore  job=${SLURM_JOB_ID:-local}  node=$(hostname)  $(date) ==="
echo "env=$ENV  trials=$TRIALS  target=$TARGET  exp=$MLFLOW_EXPERIMENT"

echo; echo "── [1/3] tune + ship production xgb tree ──────────────────────────"
"$PY" sweep_delta.py --model xgb --trials "$TRIALS" --target "$TARGET"

echo; echo "── [2/3] model exploration (zoo + uncertainty + classification) ───"
"$PY" explore_models.py

echo; echo "── [3/3] learning curve (tree vs GNN) ─────────────────────────────"
"$PY" learning_curve.py --model xgb

echo; echo "=== done $(date) ==="
echo "Artifacts: runs/models/ (shipped tree), runs/data/{model_exploration,learning_curve}.json,"
echo "           runs/figs/{model_zoo,learning_curve}.png ; MLflow exp '$MLFLOW_EXPERIMENT'"
