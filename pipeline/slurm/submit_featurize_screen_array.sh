#!/bin/bash
#SBATCH --job-name=feat_screen
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/screen_%A_%a.out
#
# QM-descriptors + xTB-ΔG screen of the WHOLE filter_v6 library (no DFT/ORCA), as a
# CHUNK-based SLURM array: ONE task = a CHUNK of molecules, processed --workers at a
# time within the 24-core task. Cheap enough (xTB + morfeus only) to chunk rather
# than one-task-per-molecule, so 221k aldehydes fit in ~1100 tasks instead of 221k.
#
# genoa bills in 1/8-node units = 24 cores ("shared by up to 8 jobs"), so cpus-per-task
# is set to 24 (not 16) to fully use the billed share — WORKERS*XTB_CORES = 24.
#
# Submit (create the log dir FIRST — SLURM won't mkdir it):
#   IN=/scratch-shared/schen3/benzoin-dg/data/library/aldehydes_clean_v5.csv
#   OUT=/scratch-shared/schen3/benzoin-dg/data/raw/screen_v5
#   N=$(($(wc -l < "$IN")-1)); CHUNK=200; NCH=$(( (N+CHUNK-1)/CHUNK )); mkdir -p "$OUT/logs"
#   sbatch --array=0-$((NCH-1))%64 --output="$OUT/logs/screen_%A_%a.out" \
#     --export=ALL,INPUT="$IN",OUTDIR="$OUT",CHUNK=$CHUNK submit_featurize_screen_array.sh
#
# Then concatenate the per-chunk CSVs (identical headers):
#   python pipeline/analyze_screen_v5.py --screen-dir "$OUT"   # see that script

PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
SCRIPT="$REPO/pipeline/compute/featurize_screen.py"
INPUT="${INPUT:?set INPUT=/abs/path.csv}"
OUTDIR="${OUTDIR:-$REPO/data/raw/screen_v5}"
CHUNK="${CHUNK:-200}"

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"

SOLVENT="${SOLVENT:-dmso}"; N_CONFS="${N_CONFS:-10}"
# 24 cores = WORKERS molecules in parallel × XTB_CORES threads each (PARALLEL_JOBS=1
# so per-molecule conformers run serially — the molecule-level pool already fills the
# share). xtb scales poorly past ~4 threads, so keep XTB_CORES small and WORKERS high.
WORKERS="${WORKERS:-12}"; XTB_CORES="${XTB_CORES:-2}"; PARALLEL_JOBS=1

ID=$SLURM_ARRAY_TASK_ID
TAG=$(printf "chunk_%04d" "$ID")
TASK_OUT="$OUTDIR/$TAG"
mkdir -p "$TASK_OUT" "$OUTDIR/logs"
[[ -f "$INPUT" ]] || { echo "ERROR: INPUT $INPUT not found"; exit 1; }

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
source "$VENV/bin/activate"
export XTBPATH="/home/schen3/xtb/share/xtb"
export OMP_NUM_THREADS=$XTB_CORES MKL_NUM_THREADS=$XTB_CORES OMP_STACKSIZE=2G KMP_STACKSIZE=2G

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

# self-heal: clear this node's dead-job orphan scratch before generating our own (the
# per-user inode quota on the shared nodespecific tree is the bottleneck at high %throttle).
# Safe against live jobs (per-id scontrol gate inside the script); timeout so a slow
# controller can't block job start.
timeout 30 bash "$REPO/pipeline/slurm/clean_node_orphans.sh" 2>/dev/null || true
WORK_DIR="${TMPDIR:-$PROJ/tmp}/screen_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT TERM INT

echo "Screen task ${SLURM_ARRAY_JOB_ID}[$ID] node=${SLURMD_NODENAME} chunk=$CHUNK $(date)"
python "$SCRIPT" \
    --input "$ROWS_CSV" --output "$TASK_OUT/features.csv" --work-dir "$WORK_DIR" \
    --xtb-bin "$XTB_BIN" --solvent "$SOLVENT" --n-confs "$N_CONFS" \
    --workers "$WORKERS" --xtb-cores "$XTB_CORES" --parallel-jobs "$PARALLEL_JOBS" \
    2>&1 | tee "$TASK_OUT/run.log"
echo "Done $TAG $(date) exit=${PIPESTATUS[0]}"
