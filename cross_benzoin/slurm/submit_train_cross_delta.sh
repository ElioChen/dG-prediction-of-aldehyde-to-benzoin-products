#!/bin/bash
#SBATCH --job-name=train_xdelta
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Generic SLURM wrapper for cross_benzoin/train_cross_delta.py -- run any
# retrain (round-N combined table, homo+cross unified table, etc.) as a
# proper batch job instead of a login-node nohup background process.
#
# Submit:
#   sbatch --output="/scratch-shared/schen3/benzoin-dg/<outdir>/train_%j.out" \
#     --export=ALL,TABLE="/abs/path/table.parquet",OUTDIR="/abs/path/outdir" \
#     cross_benzoin/slurm/submit_train_cross_delta.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
TABLE="${TABLE:?set TABLE=/abs/path/to/table.parquet}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/path/to/outdir}"
FOLDS="${FOLDS:-5}"
REPEATS="${REPEATS:-20}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
mkdir -p "$OUTDIR"
echo "train_cross_delta table=$TABLE outdir=$OUTDIR node=${SLURMD_NODENAME} $(date)"
cd "$REPO"
$PY -u cross_benzoin/train_cross_delta.py --table "$TABLE" --outdir "$OUTDIR" \
    --folds "$FOLDS" --repeats "$REPEATS"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
