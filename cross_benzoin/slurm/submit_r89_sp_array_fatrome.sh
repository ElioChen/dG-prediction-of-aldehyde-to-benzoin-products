#!/bin/bash
#SBATCH --job-name=r89_sp
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/r89_sp_%A_%a.out
#
# Rounds 8-9 DFT-label recovery -- the E_orca_SP leg. CHUNK-based array over a flat
# (id, xyz_path) work list produced by cross_benzoin/build_r89_geom_lists.py.
# orca_sp_from_geomlist.py is resume-safe (flushes each row, skips ids already in its
# own out-csv), so re-invoking a finished chunk is a fast no-op.
#
# Submit (products leg):
#   REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
#   for r in 8 9; do
#     LIST=$REPO/data/cross_benzoin/cross_round${r}_recover/product_geom_list.csv
#     OUT=$REPO/data/raw/dft_sp_cross/cross_round${r}/sp_products
#     N=$(( $(wc -l < "$LIST") - 1 )); CHUNK=100; NT=$(( (N+CHUNK-1)/CHUNK ))
#     mkdir -p "$OUT/logs"
#     sbatch --array=0-$((NT-1))%40 --output="$OUT/logs/%A_%a.out" \
#       --export=ALL,REPO="$REPO",GEOM_LIST="$LIST",OUTDIR="$OUT",CHUNK=$CHUNK \
#       cross_benzoin/slurm/submit_r89_sp_array.sh
#   done
# Aldehyde leg: same, with GEOM_LIST=.../r89_aldehyde_recover/aldehyde_geom_list.csv
# and OUTDIR=.../dft_sp_cross/r89_aldehyde_sp .
#
# Pre-launch smoke (no SLURM, 3 real ORCA SPs):
#   $PY cross_benzoin/orca_sp_from_geomlist.py --geom-list "$LIST" \
#       --out-csv /tmp/r89_smoke.csv --smoke --workers 3
#
set -o pipefail   # NOT -u: source/module reference unset vars

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/nhc-workflow/bin/python}"
GEOM_LIST="${GEOM_LIST:?set GEOM_LIST=/abs/geom_list.csv}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/output/dir}"
CHUNK="${CHUNK:-100}"; WORKERS="${WORKERS:-48}"; MAXCORE="${MAXCORE:-1500}"; TIMEOUT="${TIMEOUT:-7200}"
ORCA_BIN="${ORCA_BIN:-/home/schen3/orca/orca}"
mkdir -p "$OUTDIR/logs"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Per-SP ORCA scratch must NOT land on the shared /gpfs/scratch1 tree (inode quota
# is account-wide; a full run of 48-worker ORCA scratch there can blow it). Pin
# TMPDIR to node-local /scratch-local so orca_sp_from_geomlist.py's mkdtemp goes
# there; SLURM epilog + the script's own rmtree clean it. See the 2026-09-04 inode
# post-mortem in RUN_LOG.
if [[ -d /scratch-local ]]; then export TMPDIR="/scratch-local/${USER}.${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}"
else export TMPDIR="/tmp/${USER}.${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}"; fi
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT TERM INT HUP QUIT

ID=${SLURM_ARRAY_TASK_ID:-0}; SKIP=$(( ID * CHUNK )); TAG=$(printf "chunk_%05d" "$ID")
echo "r89_sp $TAG skip=$SKIP chunk=$CHUNK workers=$WORKERS list=$GEOM_LIST node=${SLURMD_NODENAME} $(date)"
$PY -u "$REPO/cross_benzoin/orca_sp_from_geomlist.py" \
    --geom-list "$GEOM_LIST" --skip "$SKIP" --max "$CHUNK" \
    --out-csv "$OUTDIR/${TAG}.csv" --workers "$WORKERS" \
    --maxcore "$MAXCORE" --timeout "$TIMEOUT" --orca-bin "$ORCA_BIN"
echo "Done $TAG $(date)"
