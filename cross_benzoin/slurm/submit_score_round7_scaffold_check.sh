#!/bin/bash
#SBATCH --job-name=score_r7_scafcheck
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Smoke-test + retrospective validation of the scoring pipeline fixed to use
# the scaffold-clean training pool + the ensemble architecture, by re-scoring
# round7's OWN candidate pool and comparing against what was actually selected
# under the old (leaky-split, single-XGB) scoring.
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "score_round7_scaffold_check node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/score_round_active_learning.py --tag cross_round7_scaffold_check \
    --candidates-path data/cross_benzoin/cross_round7/cross_round7_features.parquet \
    --train-table data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_clean_train.parquet \
    --feature-list data/cross_benzoin/cross_round7/scaffold_disjoint_v1/models/feature_list.json \
    --model ensemble --n-boot 20 --n-select 900 --seed 42
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
