#!/bin/bash
#SBATCH --job-name=orca_lab
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/raw/labels/logs/lab_%A_%a.out
#
# SLURM-array ΔG labeling: ONE molecule per array task, each with its own 32-core
# allocation running PARALLEL ORCA. SLURM packs ~4 tasks per 128-core node and
# spreads across the cluster, so large benzoin single-points get dedicated cores
# and never hit the per-call timeout. A failed/slow molecule is isolated to its
# own task and can be resubmitted alone.
#
# Submit with the array range matched to the input row count, e.g.:
#   N=$(($(wc -l < INPUT.csv)-1))
#   sbatch --array=0-$((N-1))%32 \
#          --export=ALL,INPUT=/abs/INPUT.csv,OUTDIR=/abs/out submit_labels_array.sh
#
# INPUT  : CSV with columns aldehyde_smiles|SMILES, index, PubChem_CID (one row/molecule)
# OUTDIR : results root; per-task output goes to $OUTDIR/mol_<id>/delta_G.csv

PROJ="/scratch-shared/schen3"
REPO="$PROJ/benzoin-dg"
SCRIPT="$REPO/pipeline/compute/thermo_orca.py"
INPUT="${INPUT:?set INPUT=/abs/path.csv}"
OUTDIR="${OUTDIR:-$REPO/data/raw/labels}"

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"
ORCA_BIN="/home/schen3/orca/orca"
SHERMO_BIN="/home/schen3/.local/bin/Shermo"

SOLVENT="dmso"; N_CONFS=10
SP_METHOD="PBE0-D4"; SP_BASIS="def2-TZVP"
ENSEMBLE_K="${ENSEMBLE_K:-1}"      # >1 = Boltzmann-average DFT ΔG over top-K conformers
# Each task = 1 molecule on 16 cores, used by BOTH phases (they run sequentially):
#   xtb numerical Hessian (ohess) -> OMP threads;  ORCA single-point -> MPI nprocs.
# The previous OMP_NUM_THREADS=1 left xtb's Hessian single-threaded — that was the
# real bottleneck (15/16 cores idle during the slow ohess), not ORCA.
ORCA_NPROCS=16
ORCA_MAXCORE=1800       # 16 x 1800 MB ~= 29 GB < 32 GB
XTB_THREADS=16          # multi-thread the xtb Hessian
MAX_ATOMS=150
WORKERS=1               # one molecule per task

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
# Multi-thread xtb (Hessian/opt); ORCA controls its own cores via %pal nprocs.
export OMP_NUM_THREADS=$XTB_THREADS OMP_STACKSIZE=2G KMP_STACKSIZE=2G
export MKL_NUM_THREADS=$XTB_THREADS
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_rmaps_base_oversubscribe=1   # --ntasks=1 gives 1 slot; allow -np 32

# Extract this task's single row into a 1-molecule CSV.
ROW_CSV="$TASK_OUT/row.csv"
python - "$INPUT" "$ID" "$ROW_CSV" <<'PY'
import sys, pandas as pd
inp, i, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
df = pd.read_csv(inp)
if i >= len(df):
    sys.exit(f"task index {i} >= rows {len(df)}")
df.iloc[[i]].to_csv(out, index=False)
print(f"task {i}: {df.iloc[i].get('index','?')}  {df.iloc[i].get('aldehyde_smiles', df.iloc[i].get('SMILES',''))}")
PY

WORK_DIR="${TMPDIR:-$PROJ/tmp}/lab_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT TERM INT

echo "Task ${SLURM_ARRAY_JOB_ID}[$ID] node=${SLURMD_NODENAME} ORCA nprocs=$ORCA_NPROCS $(date)"

SHERMO_ARGS=""
[[ -n "$SHERMO_BIN" && -x "$SHERMO_BIN" ]] && SHERMO_ARGS="--shermo-bin $SHERMO_BIN"

python "$SCRIPT" \
    --input       "$ROW_CSV"    \
    --output-dir  "$TASK_OUT"   \
    --xtb-bin     "$XTB_BIN"    \
    --orca-bin    "$ORCA_BIN"   \
    --solvent     "$SOLVENT"    \
    --n-confs     "$N_CONFS"    \
    --workers     "$WORKERS"    \
    --sp-method   "$SP_METHOD"  --sp-basis "$SP_BASIS" \
    --orca-nprocs "$ORCA_NPROCS" --orca-maxcore "$ORCA_MAXCORE" \
    --max-atoms   "$MAX_ATOMS"  --ensemble-k "$ENSEMBLE_K" \
    $SHERMO_ARGS                \
    2>&1 | tee "$TASK_OUT/run.log"

echo "Done $TAG $(date) exit=${PIPESTATUS[0]}"
