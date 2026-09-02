#!/bin/bash
#SBATCH --job-name=relabel_8r
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/nhc-workflow/bin/python}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "relabel_8rounds node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/relabel_scaffold_split_8rounds.py
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
