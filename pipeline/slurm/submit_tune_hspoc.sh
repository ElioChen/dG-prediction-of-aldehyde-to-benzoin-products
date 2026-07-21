#!/bin/bash
#SBATCH --job-name=tune_hspoc
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/tune_hspoc_%A_%a.out
#SBATCH --array=0-1
#
# Phase-1 next-step #3 hyperparameter-tuning sub-item (pipeline/bde/tune_hspoc_xgb.py):
# molecule-grouped RandomizedSearchCV vs today's default XGB config, aldehyde formyl C-H
# (array idx 0) and product ketC-carbC (array idx 1) H-SPOC. n_iter/cv_folds trimmed from
# the script's own defaults (40/4 -> 25/3) to keep worst-case wall time inside 6h.
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_lite"
BDE="$REPO/pipeline/bde"
mkdir -p "$REPO/runs/logs"

WHICH=(aldehydes products)
W="${WHICH[$SLURM_ARRAY_TASK_ID]}"

echo "H-SPOC hyperparameter tuning ($W) $(date)"
$ENV/bin/python -u "$BDE/tune_hspoc_xgb.py" \
    --which "$W" --n-iter 25 --cv-folds 3 \
    --out "$REPO/runs/logs/tune_hspoc_${W}_result.json"
echo "$W done $(date) exit=$?"
