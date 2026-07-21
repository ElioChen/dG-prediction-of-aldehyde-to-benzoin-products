#!/bin/bash
#SBATCH --job-name=dft_bde_pilot
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/dft_bde_pilot_%j.out
#
# DFT (r2SCAN-3c) ground-truth arbitration for the aldehyde formyl C-H BDE
# (pipeline/bde/dft_bde_pilot.py) -- see that script's docstring for the full
# motivation/method. cpus-per-task=1 matches this project's own ORCA convention
# (--orca-nprocs 1 everywhere else) -- avoids needing `module load 2023` for mpirun.
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_lite"
mkdir -p "$REPO/runs/logs"

N="${N:-25}"
WORKDIR="${WORKDIR:-/scratch-local/$SLURM_JOB_ID/dft_bde_pilot}"
mkdir -p "$WORKDIR"

echo "DFT BDE arbitration pilot (n=$N) $(date)"
$ENV/bin/python -u "$REPO/pipeline/bde/dft_bde_pilot.py" \
    --n "$N" --work-dir "$WORKDIR" \
    --out "$REPO/data/cross_benzoin/homo_v6/dft_bde_pilot.csv"
echo "done $(date) exit=$?"
