#!/bin/bash
#SBATCH --job-name=bz_train
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/train_%j.out
#
# Δ-learning model training as a SLURM job (not a background process), so progress
# is visible via `squeue` / `tail -f runs/logs/train_<jobid>.out` and inputs/outputs
# are explicit. Runs the Optuna sweep, then assembles the winner + AD reference
# into the package's src/benzoin_dG/models/.
#
#   sbatch pipeline/slurm/submit_train.sh                 # 60-trial sweep (default)
#   sbatch --export=ALL,TRIALS=100 pipeline/slurm/submit_train.sh
#
# Inputs  : data/descriptors/chunk_*/descriptors.csv , data/labels/chunk_*/delta_G.csv
# Outputs : runs/models/{delta_model.joblib,feature_list.json,metadata.json}
#           src/benzoin_dG/models/  (shipped model + ad_reference.npz)
#           mlflow.db  (sqlite tracking; view with `mlflow ui`)

set -euo pipefail
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"   # has rdkit+xgboost+optuna+mlflow+shap
TRIALS="${TRIALS:-120}"
MODEL="${MODEL:-all}"
TARGET="${TARGET:-dG_orca_kcal}"
PARQUET="${PARQUET:-$REPO/data/featurize.parquet}"   # label version to train on
# Versioned experiment so MLflow comparisons aren't contaminated across dataset versions.
export MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-benzoin_delta_dG_472}"

mkdir -p "$REPO/runs/logs"
cd "$REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # let sklearn/xgboost manage cores via n_jobs

echo "=== Δ-learning training  job=${SLURM_JOB_ID:-local}  node=$(hostname)  $(date) ==="
echo "Config : trials=$TRIALS model=$MODEL target=$TARGET experiment=$MLFLOW_EXPERIMENT"
echo

echo "Parquet: $PARQUET"
echo "── [1/2] Optuna sweep ─────────────────────────────────────────────"
"$PY" pipeline/sweep_delta.py --trials "$TRIALS" --model "$MODEL" --target "$TARGET" \
    --parquet "$PARQUET"

echo
echo "── [2/2] Assemble shipped model + AD reference ────────────────────"
"$PY" pipeline/assemble_model.py --target "$TARGET" --parquet "$PARQUET"

echo
echo "=== done $(date) ==="
echo "Shipped model : $REPO/src/benzoin_dG/models/"
echo "MLflow UI     : mlflow ui --backend-store-uri sqlite:///$REPO/mlflow.db"
