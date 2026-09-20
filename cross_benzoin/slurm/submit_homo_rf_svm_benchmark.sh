#!/bin/bash
#SBATCH --job-name=homo_rf_svm_bench
#SBATCH --partition=fat_genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/homo_rf_svm_bench_%j.out
#
# 2026-09-20 (user: benchmark XGB against other mainstream tabular models):
# Random Forest + Linear SVR (full train set) + RBF SVR (20k subsample,
# RBF doesn't scale to 132k rows) on the homo full-library table, same
# Delta-learning setup/split/features as the champion single-XGB.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
echo "homo_rf_svm_bench node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/train_homo_rf_svm_benchmark.py
echo "Done $(date) exit=$?"
