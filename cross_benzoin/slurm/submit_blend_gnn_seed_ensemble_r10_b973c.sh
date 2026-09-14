#!/bin/bash
#SBATCH --job-name=blend_gnn_r10b973c
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/blend_gnn_r10b973c_%j.out
#
# DRAIN_RUNBOOK.md step 5, b973c variant.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CB_TARGET_COL=dG_r2scan_kcal CB_BASELINE_COL=dG_b973c_kcal
echo "blend_gnn_r10b973c node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/blend_gnn_seed_ensemble_r10_b973c.py
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
