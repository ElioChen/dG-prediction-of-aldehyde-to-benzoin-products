#!/bin/bash
#SBATCH --job-name=dspoc_bde
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/dspoc_%x_%j.out
#
# B3 D-SPOC baseline (pipeline/bde/train_dspoc_baseline.py), on dedicated SLURM CPU
# rather than the shared login node -- a local run of this (217-descriptor variant)
# died silently after >1h with no progress, most likely starved by other users'
# interactive processes on the shared login node it was launched from.
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_lite"
mkdir -p "$REPO/runs/logs"

WHICH="${WHICH:-aldehydes}"
FULL="${FULL:-}"
OUT="${OUT:-$REPO/runs/logs/dspoc_${WHICH}_result.json}"

echo "D-SPOC baseline ($WHICH, full=$FULL) $(date)"
$ENV/bin/python -u "$REPO/pipeline/bde/train_dspoc_baseline.py" \
    --which "$WHICH" $FULL \
    --alfabet-csv "$REPO/data/cross_benzoin/homo_v6/${WHICH}_bde_alfabet.csv" \
    --out "$OUT"
echo "done $(date) exit=$?"
