#!/bin/bash
#SBATCH --job-name=lc_ensemble
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# SLURM wrapper for cross_benzoin/learning_curve_check_ensemble.py -- the
# decisive learning-curve check that holds the CURRENT production architecture
# (mordred features + MLP+XGB ensemble) fixed while varying data fraction,
# unlike the older learning_curve_check.py (fixed single-XGB, no mordred).
#
# Submit:
#   sbatch --output="/scratch-shared/schen3/benzoin-dg/<outdir>/lc_%j.out" \
#     --export=ALL,TABLE="/abs/path/to/table.parquet" \
#     cross_benzoin/slurm/submit_learning_curve_ensemble.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
TABLE="${TABLE:?set TABLE=/abs/path/to/table.parquet}"
REPEATS="${REPEATS:-5}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "learning_curve_check_ensemble table=$TABLE node=${SLURMD_NODENAME} $(date)"
cd "$REPO"
$PY -u cross_benzoin/learning_curve_check_ensemble.py --table "$TABLE" --repeats "$REPEATS"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
