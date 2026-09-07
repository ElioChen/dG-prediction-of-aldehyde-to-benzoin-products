#!/bin/bash
#SBATCH --job-name=ald_gxtb
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#
# Cheap G_gxtb recompute for the ~207k aldehydes the 09-02 BDE featurize rebuild left
# without a whole-molecule g-xTB free energy (2 fast SPs on the already-optimized,
# already-archived geometry -- NOT a re-optimization). See
# pipeline/bde/recompute_aldehyde_gxtb.py for the method and
# memory predict-dg-g-gxtb-regression-fixed / RUN_LOG_20260903.md 09-07 for why.
# Pilot: chunk_0000 full (97/97 ok, ~0.54s/molecule, 0.10 kcal/mol vs the old 42k
# library's cached G_gxtb for the one overlapping molecule) -- see RUN_LOG 09-07.
#
# genoa is fully drained/down as of 09-07 (sinfo) -- this runs on rome instead.
#
# Submit (array index = position in the chunk-dir manifest, gaps-safe):
#   REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
#   OUT=$REPO/data/cross_benzoin/bde_homo_product_featurize_20260902
#   ls -d "$OUT"/chunk_* | sort > /tmp/gxtb_manifest.txt
#   N=$(wc -l < /tmp/gxtb_manifest.txt)
#   sbatch --array=0-$((N-1))%20 pipeline/slurm/submit_recompute_aldehyde_gxtb.sh /tmp/gxtb_manifest.txt
#
set -eo pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
MANIFEST="${1:?usage: sbatch ... submit_recompute_aldehyde_gxtb.sh <manifest.txt>}"
ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
CHUNK_DIR=$(sed -n "$((ID+1))p" "$MANIFEST")
[[ -n "$CHUNK_DIR" && -d "$CHUNK_DIR" ]] || { echo "ERROR: no chunk dir at manifest line $((ID+1))"; exit 1; }
TAG=$(basename "$CHUNK_DIR")
OUT_CSV="$CHUNK_DIR/aldehydes_gxtb.csv"

VENV="${VENV:-/home/schen3/venv/nhc-workflow}"
source /etc/profile 2>/dev/null || true
module load 2023 2>/dev/null || true
source "$VENV/bin/activate" || { echo "ERROR: venv activate failed: $VENV"; exit 3; }
export XTBPATH="/home/schen3/xtb/share/xtb"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# resume guard
if [[ -f "$OUT_CSV" ]]; then
    echo "$TAG already has aldehydes_gxtb.csv - skip"
    exit 0
fi

LOCAL_BASE="/scratch-local/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"
if [[ -d /scratch-local ]]; then export TMPDIR="$LOCAL_BASE"; else export TMPDIR="/tmp/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"; fi
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT TERM INT

echo "ald_gxtb ${SLURM_ARRAY_JOB_ID:-nojob}[$ID] $TAG node=${SLURMD_NODENAME:-?} $(date)"
cd "$REPO"
python pipeline/bde/recompute_aldehyde_gxtb.py \
    --chunk-dir "$CHUNK_DIR" --out "$OUT_CSV" --timeout 300
RC=$?
echo "Done $TAG $(date) exit=$RC"
exit "$RC"
