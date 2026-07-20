#!/bin/bash
#SBATCH --job-name=dft_bde_pilot_n100
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/dft_bde_pilot_n100_%A_%a.out
#SBATCH --array=0-19
#
# Scaled-up DFT arbitration pilot (Phase-1 next-step #3, 2026-07-15): n=25 -> n=100,
# split into 20 shards of 5 molecules each (~7 CPU-min/molecule observed at n=25, so
# ~35min/shard typical, 2.5h budget for slower outliers) -- see dft_bde_pilot.py
# docstring "Scale-up" section. Same deliberately BDE-range-spanning sample (recomputed
# identically in every shard from the same --n/--seed, then chunked by --shard-idx), same
# ORCA r2SCAN-3c/CPCM(DMSO) recipe as the original n=25 run. Combine with
# merge_dft_bde_pilot_shards.py (chained via --dependency=afterany on this array's job id).
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_lite"
mkdir -p "$REPO/runs/logs"

N_TOTAL=100
N_SHARDS=20
WORKDIR="/scratch-local/$SLURM_JOB_ID/dft_bde_pilot"
mkdir -p "$WORKDIR"

echo "DFT BDE arbitration pilot shard $SLURM_ARRAY_TASK_ID/$N_SHARDS (n=$N_TOTAL) $(date)"
$ENV/bin/python -u "$REPO/pipeline/bde/dft_bde_pilot.py" \
    --n "$N_TOTAL" --n-shards "$N_SHARDS" --shard-idx "$SLURM_ARRAY_TASK_ID" \
    --work-dir "$WORKDIR" \
    --out "$REPO/data/cross_benzoin/homo_v6/dft_bde_pilot_n100_shard${SLURM_ARRAY_TASK_ID}.csv"
echo "shard $SLURM_ARRAY_TASK_ID done $(date) exit=$?"
