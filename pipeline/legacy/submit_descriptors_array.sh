#!/bin/bash
#SBATCH --job-name=desc_arr
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/raw/descriptors/logs/desc_%A_%a.out
#
# SLURM-array QM-only descriptor computation (xTB + morfeus + Multiwfn; Level 0
# RDKit-2D block removed — see ald_descriptors_qm.py). Each task
# handles a CHUNK of rows (descriptors are cheap, so per-chunk not per-molecule)
# and runs with multi-threaded xTB. Submit with the array range = #chunks, e.g.:
#   N=$(($(wc -l < INPUT.csv)-1)); CHUNK=30; NCH=$(( (N+CHUNK-1)/CHUNK ))
#   sbatch --array=0-$((NCH-1)) \
#     --export=ALL,INPUT=/abs/INPUT.csv,OUTDIR=/abs/out,CHUNK=$CHUNK submit_descriptors_array.sh

PROJ="/scratch-shared/schen3"
REPO="$PROJ/benzoin-dg"
SCRIPT="$REPO/pipeline/compute/ald_descriptors_qm.py"
INPUT="${INPUT:?set INPUT=/abs/path.csv}"
OUTDIR="${OUTDIR:-$REPO/data/raw/descriptors}"
CHUNK="${CHUNK:-30}"

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"
MWF_BIN="/home/schen3/mutiwfn/Multiwfn_noGUI"

ID=$SLURM_ARRAY_TASK_ID
TAG=$(printf "chunk_%03d" "$ID")
TASK_OUT="$OUTDIR/$TAG"
mkdir -p "$TASK_OUT" "$OUTDIR/logs"
[[ -f "$INPUT" ]] || { echo "ERROR: INPUT $INPUT not found"; exit 1; }

source "$VENV/bin/activate"
export XTBPATH="/home/schen3/xtb/share/xtb"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_STACKSIZE=2G

# Slice this task's chunk of rows into its own CSV.
ROWS_CSV="$TASK_OUT/rows.csv"
python - "$INPUT" "$ID" "$CHUNK" "$ROWS_CSV" <<'PY'
import sys, pandas as pd
inp, i, chunk, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
df = pd.read_csv(inp)
lo, hi = i*chunk, min((i+1)*chunk, len(df))
if lo >= len(df):
    open(out, "w").write(",".join(df.columns)+"\n"); sys.exit(0)
df.iloc[lo:hi].to_csv(out, index=False)
print(f"chunk {i}: rows {lo}:{hi} ({hi-lo} molecules)")
PY

WORK_DIR="${TMPDIR:-$PROJ/tmp}/desc_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT TERM INT

echo "Desc task ${SLURM_ARRAY_JOB_ID}[$ID] node=${SLURMD_NODENAME} $(date)"
python "$SCRIPT" \
    --input    "$ROWS_CSV" \
    --output   "$TASK_OUT/descriptors.csv" \
    --work-dir "$WORK_DIR" \
    --xtb-bin  "$XTB_BIN" \
    --n-confs  5 \
    --multiwfn --multiwfn-bin "$MWF_BIN"
EXIT=$?
echo "Done $TAG $(date) exit=$EXIT"
exit $EXIT
