#!/bin/bash
#SBATCH --job-name=bf_mwf_cb
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#
# Backfill ADCH/QTAIM Multiwfn descriptors onto an existing cb_featurize chunk dir
# (aldehydes.csv + products.csv), reusing the geometries already saved there
# (xyz_file column) -- NO re-optimization / conformer search, just xTB-molden SP +
# Multiwfn per molecule. ONE array task = ONE existing chunk_XXXX dir. genoa 24-core
# unit. Mirrors submit_backfill_mwf_array.sh (screen_v6, aldehyde-only); this one also
# does the product side via backfill_multiwfn_product.py (core-anchored, not CHO-anchored).
#
# Submit (LIB = a cb_featurize output dir with chunk_* subdirs, e.g. data/cross_benzoin/homo_v6):
#   LIB=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6
#   NCH=$(ls -d "$LIB"/chunk_* | wc -l); mkdir -p "$LIB/logs_mwf"
#   sbatch --array=0-$((NCH-1))%64 --output="$LIB/logs_mwf/bf_%A_%a.out" \
#     --export=ALL,LIB="$LIB" cross_benzoin/slurm/submit_backfill_mwf_cb_array.sh
#
# Then merge ald_multiwfn.csv / prod_multiwfn.csv into aldehydes.csv / products.csv on id.
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
ALD_SCRIPT="$REPO/pipeline/compute/backfill_multiwfn.py"
PROD_SCRIPT="$REPO/pipeline/compute/backfill_multiwfn_product.py"
LIB="${LIB:?set LIB=/abs/path/to/cb_featurize/output/dir}"

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"
MWF_BIN="/home/schen3/mutiwfn/Multiwfn_noGUI"
WORKERS="${WORKERS:-12}"              # default 12x2; pass WORKERS=24 XTB_THREADS=1
XTB_THREADS="${XTB_THREADS:-2}"       # 24-core node unit: workers x threads = 24

ID=$SLURM_ARRAY_TASK_ID
TAG=$(printf "chunk_%04d" "$ID")
CHUNK_DIR="$LIB/$TAG"
ALD_IN="$CHUNK_DIR/aldehydes.csv"
PROD_IN="$CHUNK_DIR/products.csv"
ALD_OUT="$CHUNK_DIR/ald_multiwfn.csv"
PROD_OUT="$CHUNK_DIR/prod_multiwfn.csv"
[[ -d "$CHUNK_DIR" ]] || { echo "skip: $CHUNK_DIR not found"; exit 0; }

_done() {  # $1=input $2=output -> 0 if output already covers input row count
    [[ -f "$2" ]] || return 1
    local nin nout
    nin=$(($(wc -l < "$1") - 1)); nout=$(($(wc -l < "$2") - 1))
    [[ "$nout" -ge "$nin" ]]
}

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
source "$VENV/bin/activate"
export XTBPATH="/home/schen3/xtb/share/xtb"
export OMP_NUM_THREADS=$XTB_THREADS MKL_NUM_THREADS=$XTB_THREADS OMP_STACKSIZE=2G KMP_STACKSIZE=2G

WORK_DIR="${TMPDIR:-$PROJ/tmp}/bf_mwf_cb_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT TERM INT

echo "Backfill MWF cb ${SLURM_ARRAY_JOB_ID}[$ID] $TAG node=${SLURMD_NODENAME} $(date)"
cd "$REPO/pipeline/compute"

if [[ -f "$ALD_IN" ]] && ! _done "$ALD_IN" "$ALD_OUT"; then
    python "$ALD_SCRIPT" --input "$ALD_IN" --output "$ALD_OUT" --work-dir "$WORK_DIR/ald" \
        --xtb-bin "$XTB_BIN" --multiwfn-bin "$MWF_BIN" --workers "$WORKERS" \
        2>&1 | tee "$CHUNK_DIR/backfill_mwf_ald.log"
else
    echo "skip ald: $TAG already done or no $ALD_IN"
fi

if [[ -f "$PROD_IN" ]] && ! _done "$PROD_IN" "$PROD_OUT"; then
    python "$PROD_SCRIPT" --input "$PROD_IN" --output "$PROD_OUT" --work-dir "$WORK_DIR/prod" \
        --xtb-bin "$XTB_BIN" --multiwfn-bin "$MWF_BIN" --workers "$WORKERS" \
        2>&1 | tee "$CHUNK_DIR/backfill_mwf_prod.log"
else
    echo "skip prod: $TAG already done or no $PROD_IN"
fi

echo "Done $TAG $(date) exit=$?"
