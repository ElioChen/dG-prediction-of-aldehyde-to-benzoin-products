#!/bin/bash
#SBATCH --job-name=score_r10
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# Score Round10's 7,995 unordered-pair candidate pool by bootstrap ensemble
# uncertainty, using the reproduced rounds1-7 table/model as the reference (the
# only training table with a surviving, verified label set as of the 2026-09
# recovery -- rounds8-9 labels have no surviving copy). Round9's champion model
# was trained on 43,367 rows including rounds8-9, but score_round_active_learning.py
# refits bootstrap resamples FROM the train table itself, so a trained model alone
# isn't enough; the table it was trained on is required and only rounds1-7's
# survives.
#
REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/nhc-workflow/bin/python}"
N_BOOT="${N_BOOT:-40}"
N_SELECT="${N_SELECT:-2000}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO" || exit 1
echo "score_round10_active_learning node=${SLURMD_NODENAME:-unknown} $(date)"
$PY -u cross_benzoin/score_round_active_learning.py \
    --tag cross_round10_fat20_stage1 \
    --candidates-path data/cross_benzoin/cross_round10_fat20_stage1/cross_round10_fat20_stage1_features_cross_round10_products_merged.parquet \
    --train-table data/cross_benzoin/cross_round7/cross_train_table_7rounds_recovered_slim260.parquet \
    --feature-list data/cross_benzoin/cross_round7/train_ensemble_7rounds_recovered_v1/models/feature_list.json \
    --model ensemble --n-boot "$N_BOOT" --n-select "$N_SELECT" --seed 42
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
