#!/bin/bash
#SBATCH --job-name=dftbde_geom
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/dftbde_geom_%A_%a.out
#SBATCH --array=0-8
#
# DFT geometry-vs-single-point decomposition of the g-xTB BDE label error (2026-07-16).
# pipeline/bde/dft_bde_geom_arbitration.py -- adds the r2SCAN-3c GEOMETRY-OPTIMIZATION leg
# the earlier SP-only pilots never had, to split the ~14.7 kcal g-xTB-vs-DFT label gap into
# an SP-method component and a GFN2-geometry component (user question, 2026-07-16). CPU/ORCA;
# will queue behind the busy NHC jobs -- that's expected, it's not urgent.
#
# 18 molecules (3 each x 5 target functional groups + 3 controls), sharded 9-way so each
# array task does ~2 molecules; each molecule = 2 DFT geometry opts (parent + acyl radical)
# + 1 DFT SP, at nprocs=1 (ORCA %pal needs mpirun, not on PATH here -- same constraint
# dft_bde_pilot.py already worked around; a first attempt at this job hardcoded --nprocs 8
# here and every ORCA call died instantly, see PROGRESS_20260714.md). Incrementally flushes
# its own out-csv (resume-safe per row); merge the 9 shard csvs afterward with a trivial
# pandas concat.
set -o pipefail   # not -u: module load / source reference unset vars

REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
OUT="$REPO/data/cross_benzoin/homo_v6/dft_bde_geom_arbitration"
mkdir -p "$OUT"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:/home/schen3/orca/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ID=${SLURM_ARRAY_TASK_ID:-0}
echo "dftbde_geom shard $ID node=${SLURMD_NODENAME} $(date)"
$PY -u "$REPO/pipeline/bde/dft_bde_geom_arbitration.py" \
    --per-group 3 --n-control 3 --nprocs 1 --timeout 21600 \
    --n-shards 9 --shard-idx "$ID" \
    --work-dir "/scratch-local/dftbde_geom_${SLURM_ARRAY_JOB_ID}_${ID}" \
    --out "$OUT/shard_${ID}.csv"
echo "shard $ID done $(date) exit=$?"
