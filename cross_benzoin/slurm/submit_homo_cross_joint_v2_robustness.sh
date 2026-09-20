#!/bin/bash
#SBATCH --job-name=homo_cross_robustness
#SBATCH --partition=fat_genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/homo_cross_robustness_%j.out
#
# 2026-09-20 (user: "同意，同步推进" -- go ahead with the robustness check on
# the Rec-2 AMBER verdict): 5-seed refit + 10k-resample holdout bootstrap.
# Lesson from earlier the same day: submit real compute to SLURM, don't run
# it as a raw background Bash process on the login node (see
# login-node-background-bash-silent-kill memory).
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "homo_cross_robustness node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/homo_cross_joint_tabular_v2_robustness.py
RC=$?
echo "Done $(date) exit=$RC"
exit $RC
