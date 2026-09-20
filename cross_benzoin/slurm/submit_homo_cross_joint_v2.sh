#!/bin/bash
#SBATCH --job-name=homo_cross_joint_v2
#SBATCH --partition=fat_genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/homo_cross_joint_v2_%j.out
#
# 2026-09-20: Rec-2 unification retrain on the real full-library homo table.
# Two earlier attempts run as raw background Bash processes on the
# interactive login node (int4) both died silently right after
# naive_merge_weighted -- no exception, no exit code, just gone -- most
# likely an interactive-node time/resource policy killing it partway through
# the heavy (n_estimators=1800) homo_only_zeroshot/finetune fits. Moving this
# to a real SLURM job instead of retrying on the login node.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "homo_cross_joint_v2 node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/homo_cross_joint_tabular_v2.py \
    --homo-table data/cross_benzoin/homo_standalone/homo_standalone_full_library_table_slim260.parquet
RC=$?
echo "Done $(date) exit=$RC"
exit $RC
