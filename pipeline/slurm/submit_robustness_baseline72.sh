#!/bin/bash
#SBATCH --job-name=robust72
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --time=02:00:00
#
# Array task ID 0-4  -> holdout reshuffle, seed = task id
# Array task ID 5-9  -> 5-fold CV, fold = task id - 5
#
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
ID=$SLURM_ARRAY_TASK_ID
if [[ $ID -lt 5 ]]; then
    echo "holdout seed=$ID $(date)"
    $PY -u "$REPO/pipeline/analysis/robustness_baseline72.py" --mode holdout --seed "$ID"
else
    FOLD=$((ID - 5))
    echo "cv fold=$FOLD $(date)"
    $PY -u "$REPO/pipeline/analysis/robustness_baseline72.py" --mode cv --fold "$FOLD" --nfolds 5
fi
echo "Done task=$ID $(date) exit=$?"
