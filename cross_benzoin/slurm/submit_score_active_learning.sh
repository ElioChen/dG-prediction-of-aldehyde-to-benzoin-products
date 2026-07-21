#!/bin/bash
#SBATCH --job-name=score_al
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Generalized SLURM wrapper for score_round_active_learning.py.
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"
TAG="${TAG:?set TAG}"
TRAIN_TABLE="${TRAIN_TABLE:?set TRAIN_TABLE}"
FEATURE_LIST="${FEATURE_LIST:?set FEATURE_LIST}"
MODEL="${MODEL:-ensemble}"
N_BOOT="${N_BOOT:-40}"
N_SELECT="${N_SELECT:-4000}"
SEED="${SEED:-42}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "score_active_learning tag=$TAG model=$MODEL node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/score_round_active_learning.py --tag "$TAG" \
    --train-table "$TRAIN_TABLE" --feature-list "$FEATURE_LIST" \
    --model "$MODEL" --n-boot "$N_BOOT" --n-select "$N_SELECT" --seed "$SEED"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
