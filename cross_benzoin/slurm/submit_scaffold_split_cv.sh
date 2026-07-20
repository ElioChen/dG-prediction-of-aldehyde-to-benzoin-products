#!/bin/bash
#SBATCH --job-name=scaffold_cv
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# SLURM wrapper for cross_benzoin/analysis/scaffold_split_cv_cross.py --
# scaffold-disjoint generalization test, mirrors homo's exp_scaffold_split.py.
#
# Submit:
#   sbatch --output="/scratch-shared/schen3/benzoin-dg/<outdir>/scaffold_%j.out" \
#     --export=ALL,TABLE="/abs/table.parquet",OUTDIR="/abs/outdir" \
#     cross_benzoin/slurm/submit_scaffold_split_cv.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
TABLE="${TABLE:?set TABLE=/abs/path/to/table.parquet}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/path/to/outdir}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
mkdir -p "$OUTDIR"
echo "scaffold_split_cv_cross table=$TABLE outdir=$OUTDIR node=${SLURMD_NODENAME} $(date)"
cd "$REPO"
$PY -u cross_benzoin/analysis/scaffold_split_cv_cross.py --table "$TABLE" --outdir "$OUTDIR"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
