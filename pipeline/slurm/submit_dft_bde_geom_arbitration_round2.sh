#!/bin/bash
#SBATCH --job-name=dftbde_geom2
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/dftbde_geom2_%A_%a.out
#SBATCH --array=0-63
#
# Round 2 of the DFT geometry-vs-single-point arbitration (2026-07-17), scaled from the
# n=18 pilot (job 24665669) to n=128 -- candidates chosen by the new active-learning
# selector (pipeline/bde/select_dft_arbitration_batch.py: functional-group stratification,
# now including selenium; farthest-point diversity in QM-descriptor space; weak
# bootstrap-ensemble uncertainty re-rank on the n=17 existing arbitration points).
# 128 molecules / 2 per shard = 64 shards, same per-molecule cost profile as round 1
# (1 CPU, 8GB mem -- ORCA %pal needs mpirun which isn't on PATH, so nprocs=1; genoa was
# chosen over gpu_h100/fat because this job is wall-clock-bound, not memory-bound: round1's
# shards used <8GB comfortably even for the largest 68-atom sulfonyl molecule, and genoa's
# queue was light (17 running/3 pending) with 95M+ SBUs of budget headroom at submit time
# per `budget-overview -p genoa`).
set -o pipefail   # not -u: module load / source reference unset vars

REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
OUT="$REPO/data/cross_benzoin/homo_v6/dft_bde_geom_arbitration_round2"
CAND="$OUT/candidates_128.csv"
mkdir -p "$OUT"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:/home/schen3/orca/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ID=${SLURM_ARRAY_TASK_ID:-0}
echo "dftbde_geom2 shard $ID node=${SLURMD_NODENAME} $(date)"
$PY -u "$REPO/pipeline/bde/dft_bde_geom_arbitration.py" \
    --candidates-csv "$CAND" --nprocs 1 --timeout 21600 \
    --n-shards 64 --shard-idx "$ID" \
    --work-dir "/scratch-local/dftbde_geom2_${SLURM_ARRAY_JOB_ID}_${ID}" \
    --out "$OUT/shard_${ID}.csv"
echo "shard $ID done $(date) exit=$?"
