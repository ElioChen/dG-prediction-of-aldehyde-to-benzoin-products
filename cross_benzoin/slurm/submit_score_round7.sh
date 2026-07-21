#!/bin/bash
#SBATCH --job-name=score_r7
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# CPU-only bootstrap-uncertainty scoring of round7's leftover screen10k reservoir
# (3,677 pairs) against the round1-6 training table. genoa (not rome) is this
# project's usual CPU partition and had only ~15 of this account's jobs queued
# at submit time (vs rome's 141, which had hit QOSMaxJobsPerUserLimit) -- not
# congested enough to justify burning a full gpu_h100 GPU allocation on a job
# that's pure CPU/sklearn/xgboost work and never touches CUDA. gpu_h100 also
# turned out to REQUIRE >=1 GPU per job (verified via `scontrol show partition
# gpu_h100`), so it's reserved for genuinely GPU-bound work (the GNN retune)
# rather than used as a blanket CPU-congestion escape valve.
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "score_round7 node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/score_round_active_learning.py --tag cross_round7 \
    --train-table data/cross_benzoin/cross_round6/cross_train_table_6rounds_mordred_slim120_matched.parquet \
    --feature-list data/cross_benzoin/cross_round6/train_6rounds_mordred_slim120_v1/models/feature_list.json \
    --n-boot 40 --n-select 3677 --seed 42
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
