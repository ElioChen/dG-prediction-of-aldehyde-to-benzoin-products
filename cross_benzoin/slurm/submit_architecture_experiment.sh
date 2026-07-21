#!/bin/bash
#SBATCH --job-name=arch_exp
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
REPO="/gpfs/scratch1/shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
TABLE="${TABLE:?set TABLE=/abs/path/to/table.parquet}"
FOLDS="${FOLDS:-5}"
REPEATS="${REPEATS:-10}"
cd "$REPO"
$PY -u cross_benzoin/architecture_ensemble_experiment.py \
    --table "$TABLE" --folds "$FOLDS" --repeats "$REPEATS"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
