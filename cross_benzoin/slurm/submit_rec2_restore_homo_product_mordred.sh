#!/bin/bash
#SBATCH --job-name=rec2_restore_pmordred
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/rec2_restore_pmordred_%j.out
#
# Restore full homo product Mordred from the 2026-07-15 home backup
# (mordred_products.tar.gz, 2196 chunks). Decompress + concat + dedup, no compute.
# genoa short borrow, CPU, <1h.
set -euo pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
cd "$REPO"
WD=/scratch-local/${USER}.${SLURM_JOB_ID:-restore}
mkdir -p "$WD" "$REPO/slurm_logs"
echo "rec2_restore_pmordred node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/rec2_restore_homo_product_mordred.py --workdir "$WD"
rm -rf "$WD" 2>/dev/null || true
echo "Done $(date)"
