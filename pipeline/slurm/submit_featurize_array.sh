#!/bin/bash
#SBATCH --job-name=featurize
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/feat_%A_%a.out
#
# Unified per-molecule featurization as a SLURM array: ONE task = ONE molecule =
# {shared aldehyde conformer search + xTB-opt} → descriptors + (ensemble) ΔG, all
# on the same geometry. Replaces the separate descriptor + label arrays and removes
# the duplicate conformer work / descriptor-vs-label geometry mismatch.
#
# LOGS: the static --output above (runs/logs/) is only a fallback. ALWAYS override
# --output to the batch's own OUTDIR/logs so each batch's SLURM logs stay with its
# data instead of piling into one shared dir. Create that dir BEFORE sbatch (SLURM
# opens the log file at job start and will NOT mkdir it):
#
#   N=$(($(wc -l < INPUT.csv)-1)); mkdir -p /abs/out/logs
#   ENSEMBLE_K=5 sbatch --array=0-$((N-1))%192 --output=/abs/out/logs/feat_%A_%a.out \
#     --export=ALL,INPUT=/abs/INPUT.csv,OUTDIR=/abs/out submit_featurize_array.sh

PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
SCRIPT="$REPO/pipeline/compute/featurize.py"
INPUT="${INPUT:?set INPUT=/abs/path.csv}"
OUTDIR="${OUTDIR:-$REPO/data/raw/featurize}"

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"
ORCA_BIN="/home/schen3/orca/orca"
MWF_BIN="/home/schen3/mutiwfn/Multiwfn_noGUI"

SOLVENT="dmso"; N_CONFS="${N_CONFS:-10}"
SP_METHOD="r2SCAN-3c"; SP_BASIS=""
MAX_ATOMS=150
ENSEMBLE_K="${ENSEMBLE_K:-3}"      # Boltzmann-average DFT ΔG over top-K conformers
# Featurize entry point — override to run a variant (e.g. featurize_funnel.py) for
# a side-by-side comparison without touching the canonical featurize.py.
SCRIPT="${FEATURIZE_SCRIPT:-$SCRIPT}"
# Within the 16-core task, run PARALLEL_JOBS conformer sub-jobs concurrently, each
# with its own xTB (XTB_CORES) + ORCA (ORCA_NPROCS). xtb scales poorly past ~4
# cores; ORCA gets 4-16. PARALLEL_JOBS x ORCA_NPROCS = 16 keeps the node full.
PARALLEL_JOBS=4; ORCA_NPROCS=4; XTB_CORES=2; ORCA_MAXCORE=3000

ID=$SLURM_ARRAY_TASK_ID
TAG=$(printf "mol_%04d" "$ID")
TASK_OUT="$OUTDIR/$TAG"
mkdir -p "$TASK_OUT" "$OUTDIR/logs"
[[ -f "$INPUT" ]] || { echo "ERROR: INPUT $INPUT not found"; exit 1; }

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
module load OpenMPI/4.1.5-GCC-12.3.0 2>/dev/null
source "$VENV/bin/activate"
export XTBPATH="/home/schen3/xtb/share/xtb"
# xtb thermo threads come from --xtb-cores (its --parallel flag); keep OMP modest
# for the descriptor xtb SP / morfeus / numpy so concurrent sub-jobs don't thrash.
export OMP_NUM_THREADS=$XTB_CORES MKL_NUM_THREADS=$XTB_CORES OMP_STACKSIZE=2G KMP_STACKSIZE=2G
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_rmaps_base_oversubscribe=1

# Extract this task's single row.
ROW_CSV="$TASK_OUT/row.csv"
python - "$INPUT" "$ID" "$ROW_CSV" <<'PY'
import sys, pandas as pd
inp, i, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
df = pd.read_csv(inp)
if i >= len(df): sys.exit(f"task {i} >= rows {len(df)}")
df.iloc[[i]].to_csv(out, index=False)
print(f"task {i}: {df.iloc[i].get('index','?')}")
PY

WORK_DIR="${TMPDIR:-$PROJ/tmp}/feat_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT TERM INT

echo "Task ${SLURM_ARRAY_JOB_ID}[$ID] node=${SLURMD_NODENAME} ensemble_k=$ENSEMBLE_K $(date)"
python "$SCRIPT" \
    --input "$ROW_CSV" --output "$TASK_OUT/features.csv" --work-dir "$WORK_DIR" \
    --xtb-bin "$XTB_BIN" --multiwfn-bin "$MWF_BIN" --orca-bin "$ORCA_BIN" \
    --solvent "$SOLVENT" --n-confs "$N_CONFS" --ensemble-k "$ENSEMBLE_K" \
    --sp-method "$SP_METHOD" --sp-basis "$SP_BASIS" \
    --orca-nprocs "$ORCA_NPROCS" --orca-maxcore "$ORCA_MAXCORE" --max-atoms "$MAX_ATOMS" \
    --xtb-cores "$XTB_CORES" --parallel-jobs "$PARALLEL_JOBS" \
    2>&1 | tee "$TASK_OUT/run.log"
echo "Done $TAG $(date) exit=${PIPESTATUS[0]}"
