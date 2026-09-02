#!/bin/bash
#SBATCH --job-name=train_r17
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#
# The reproduction test for the 2026-09 purge recovery: retrain the exact 7-round
# architecture on the rebuilt table and compare against the surviving original run
# (cross_round7/train_ensemble_7rounds_slim120_v1/models/metadata.json):
#
#     model     mlp128_64+xgb_d5+xgb_d7_ensemble
#     n_samples 32,456   n_pairs 16,241   n_features 260
#     CV(5-fold x 10 repeats)  MAE 1.8830  RMSE 3.7520  R2 0.5629
#     g-xTB baseline           MAE 3.9050
#
# The rebuilt table has 32,418 rows / 16,222 pairs -- 38 rows short (0.12%), lost to
# a handful of aldehydes whose recompute errored. Same seed (42), folds and repeats,
# and the table is pre-pruned to the original run's own feature_list.json so
# _feature_blocks reproduces its 260-feature selection exactly.
#
REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/nhc-workflow/bin/python}"
TABLE="${TABLE:-$REPO/data/cross_benzoin/cross_round7/cross_train_table_7rounds_recovered_slim260.parquet}"
OUTDIR="${OUTDIR:-$REPO/data/cross_benzoin/cross_round7/train_ensemble_7rounds_recovered_v1}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$REPO" || exit 1
echo "train_ensemble_r17_recovered node=${SLURMD_NODENAME:-unknown} cpus=${SLURM_CPUS_PER_TASK:-?} $(date)"
echo "table=$TABLE outdir=$OUTDIR"
$PY -u cross_benzoin/train_cross_ensemble.py \
    --table "$TABLE" --outdir "$OUTDIR" --folds 5 --repeats 10 --seed 42
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
