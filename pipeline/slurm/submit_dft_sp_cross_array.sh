#!/bin/bash
#SBATCH --job-name=dftsp_cross_arr
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_cross/logs/dftsp_cross_%A_%a.out
#
# CHUNK-based ARRAY version of submit_dft_sp_cross.sh, for cross-benzoin product sets big
# enough that single-node 48-way parallelism is the bottleneck (e.g. cross_pilot_v2's 4200
# rows -- cross_pilot_v1's 598 was small enough that the single-task version was fine, but
# products here vary a lot in size/flexibility and don't parallelize predictably; spreading
# across many nodes gives much better wall-clock and per-chunk fault isolation than one node
# doing everything).
#
# pipeline/compute/dft_sp_cross_from_geom.py is internally resume-safe (flushes each row,
# skips ids already in its own out-csv), so this array does NOT need its own "already done"
# file-exists gate -- re-invoking on an already-complete chunk is a fast no-op. It DOES use
# --manifest-cache so only the FIRST task to reach it pays for rebuilding the id_map/thermal/
# DFT-SP-energy lookups (~220k aldehyde rows each); the write is atomic (temp+os.replace) so
# concurrent array tasks racing to build it can't corrupt it.
#
# Submit (products CSV = a cb_featurize.py-produced/consolidated products.csv):
#   PRODUCTS=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/cross_pilot_v2/cross_pilot_v2_products.csv
#   OUT=/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_cross/cross_pilot_v2
#   CHUNK=100; NT=$(( (4200+CHUNK-1)/CHUNK ))   # or count real manifest rows first, see below
#   mkdir -p "$OUT/logs"
#   sbatch --array=0-$((NT-1))%42 --output="$OUT/logs/dftsp_cross_%A_%a.out" \
#     --export=ALL,PRODUCTS="$PRODUCTS",OUTDIR="$OUT",CHUNK=$CHUNK submit_dft_sp_cross_array.sh
#
# PRE-LAUNCH smoke test (no SLURM, ~3 real ORCA SPs, run from a login/compute shell):
#   python pipeline/compute/dft_sp_cross_from_geom.py --products-csv "$PRODUCTS" \
#       --out-csv /tmp/smoke.csv --smoke --workers 3
#
set -o pipefail   # NOT -u: `source /etc/profile` / `module load` reference unset vars and crash under -u
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
PRODUCTS="${PRODUCTS:?set PRODUCTS=/abs/products.csv}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/output/dir}"
CHUNK="${CHUNK:-100}"; WORKERS="${WORKERS:-48}"; MAXCORE="${MAXCORE:-1500}"; TIMEOUT="${TIMEOUT:-7200}"
ORCA_BIN="/home/schen3/orca/orca"
MANIFEST_CACHE="$OUTDIR/manifest_cache.csv"
mkdir -p "$OUTDIR/logs"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ID=${SLURM_ARRAY_TASK_ID:-0}; SKIP=$(( ID * CHUNK )); TAG=$(printf "chunk_%05d" "$ID")

echo "dftsp_cross_arr $TAG skip=$SKIP chunk=$CHUNK workers=$WORKERS node=${SLURMD_NODENAME} $(date)"
$PY -u "$REPO/pipeline/compute/dft_sp_cross_from_geom.py" \
    --products-csv "$PRODUCTS" --manifest-cache "$MANIFEST_CACHE" \
    --skip "$SKIP" --max "$CHUNK" \
    --out-csv "$OUTDIR/${TAG}.csv" --workers "$WORKERS" --maxcore "$MAXCORE" \
    --timeout "$TIMEOUT" --orca-bin "$ORCA_BIN"
echo "Done $TAG $(date)"
