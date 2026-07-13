#!/bin/bash
#SBATCH --job-name=bf_mwf
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#
# Backfill aldehyde Multiwfn (ADCH/QTAIM) onto an existing CHUNKED screen output
# (e.g. screen_v6). ONE array task = ONE existing chunk: it reads that chunk's
# features.csv and reuses the geometries already saved there (xyz_file) — NO
# re-optimization, just xTB-molden single points + Multiwfn. genoa 24-core unit.
#
# Submit (NCH = number of existing chunk_* dirs):
#   SCREEN=/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6
#   NCH=$(ls -d "$SCREEN"/chunk_* | wc -l); mkdir -p "$SCREEN/logs_mwf"
#   sbatch --array=0-$((NCH-1))%64 --output="$SCREEN/logs_mwf/bf_%A_%a.out" \
#     --export=ALL,SCREEN_DIR="$SCREEN" submit_backfill_mwf_array.sh
#
# Then merge: per chunk join chunk_XXXX/ald_multiwfn.csv into features.csv on `index`
# (see merge_multiwfn.py).
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
SCRIPT="$REPO/pipeline/compute/backfill_multiwfn.py"
SCREEN_DIR="${SCREEN_DIR:?set SCREEN_DIR=/abs/screen_v6}"

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"
MWF_BIN="/home/schen3/mutiwfn/Multiwfn_noGUI"
WORKERS="${WORKERS:-12}"              # default 12x2; pass WORKERS=24 XTB_THREADS=1
XTB_THREADS="${XTB_THREADS:-2}"      # 24-core node unit: workers x threads = 24

ID=$SLURM_ARRAY_TASK_ID
TAG=$(printf "chunk_%04d" "$ID")
CHUNK_DIR="$SCREEN_DIR/$TAG"
INPUT="$CHUNK_DIR/features.csv"
OUTPUT="$CHUNK_DIR/ald_multiwfn.csv"
[[ -f "$INPUT" ]] || { echo "skip: $INPUT not found"; exit 0; }
# Idempotency: skip a chunk already backfilled to full row count (safe re-submits).
if [[ -f "$OUTPUT" ]]; then
    nin=$(($(wc -l < "$INPUT") - 1)); nout=$(($(wc -l < "$OUTPUT") - 1))
    [[ "$nout" -ge "$nin" ]] && { echo "skip: $TAG already done ($nout/$nin)"; exit 0; }
fi

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
source "$VENV/bin/activate"
export XTBPATH="/home/schen3/xtb/share/xtb"
export OMP_NUM_THREADS=$XTB_THREADS MKL_NUM_THREADS=$XTB_THREADS OMP_STACKSIZE=2G KMP_STACKSIZE=2G

WORK_DIR="${TMPDIR:-$PROJ/tmp}/bf_mwf_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT TERM INT

echo "Backfill MWF ${SLURM_ARRAY_JOB_ID}[$ID] $TAG node=${SLURMD_NODENAME} $(date)"
cd "$REPO/pipeline/compute"
python "$SCRIPT" \
    --input "$INPUT" --output "$OUTPUT" --work-dir "$WORK_DIR" \
    --xtb-bin "$XTB_BIN" --multiwfn-bin "$MWF_BIN" --workers "$WORKERS" \
    2>&1 | tee "$CHUNK_DIR/backfill_mwf.log"
echo "Done $TAG $(date) exit=${PIPESTATUS[0]} -> $OUTPUT"
