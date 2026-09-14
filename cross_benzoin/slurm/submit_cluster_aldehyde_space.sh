#!/bin/bash
#SBATCH --job-name=cluster_ald
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/cluster_ald_%j.out
#
# cross_benzoin/cluster_aldehyde_space.py -- run on a real allocation, not the
# interactive session (which pins OMP_NUM_THREADS=1/MKL_NUM_THREADS=1 for
# shared-node politeness and made the XGB fits ~10x slower than they should
# be, 2026-09-14). K, --reuse-clusters etc. passed through via $ARGS.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ARGS="${ARGS:---k 150 --reuse-clusters --workers 8}"
echo "cluster_aldehyde_space args=$ARGS node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/cluster_aldehyde_space.py $ARGS
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
