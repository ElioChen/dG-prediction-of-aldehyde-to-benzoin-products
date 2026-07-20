#!/bin/bash
#SBATCH --job-name=train_scafdis
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# SLURM wrapper for cross_benzoin/train_scaffold_disjoint.py -- champion +
# ensemble retrain/eval on the corrected scaffold-disjoint split (replaces
# the leaky candidates_v3 molecule-level split).
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
TABLE="${TABLE:?set TABLE=/abs/path/to/table.parquet}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/path/to/outdir}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
mkdir -p "$OUTDIR"
echo "train_scaffold_disjoint table=$TABLE outdir=$OUTDIR node=${SLURMD_NODENAME} $(date)"
cd "$REPO"
$PY -u cross_benzoin/train_scaffold_disjoint.py --table "$TABLE" --outdir "$OUTDIR"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
