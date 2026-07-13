#!/bin/bash
#SBATCH --job-name=gxtb_base
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#
# g-xTB baseline ΔG (COSMO/DMSO) on the in-scope (aromatic) labeled benzoin set,
# on the SAME funnel_v3 (GFN2-ohess) geometry the DFT label sits on. One array task
# = a CHUNK-slice of molecules (via --skip/--max). Resume-safe: skips a chunk whose
# output CSV already exists. genoa bills 24-core units -> cpus-per-task=24, workers 24.
#
#   SEL=.../data/raw/gxtb_baseline/input_inscope.csv
#   N=$(($(wc -l < "$SEL")-1)); CH=50; NT=$(( (N+CH-1)/CH ))
#   sbatch --array=0-$((NT-1))%64 submit_gxtb_baseline.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SCRIPT="$REPO/pipeline/compute/gxtb_baseline.py"
SEL="$REPO/data/raw/gxtb_baseline/input_inscope.csv"
RESULTS="$REPO/data/raw/gxtb_baseline/chunks"
CHUNK="${CHUNK:-50}"
WORKERS="${WORKERS:-24}"

export XTB_BIN="/home/schen3/xtb/bin/xtb"
export GXTB_BIN="/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb"
export GXTB_SOLV="cosmo dmso"
export OMP_NUM_THREADS=1            # funnel/ohess are single-core; parallel across mols
mkdir -p "$RESULTS"

source /etc/profile 2>/dev/null
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SKIP=$(( TASK * CHUNK ))
OUT="$RESULTS/chunk_$(printf '%05d' "$TASK").csv"
if [[ -s "$OUT" ]]; then
  echo "chunk $TASK already done ($OUT) — skipping"; exit 0
fi
bash "$REPO/pipeline/slurm/clean_node_orphans.sh" 2>/dev/null || true   # self-heal: timeout protects against SLURM controller throttling
SCRATCH="/scratch-local/$USER.$SLURM_JOB_ID.$TASK"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

echo "task=$TASK skip=$SKIP chunk=$CHUNK workers=$WORKERS out=$OUT"
"$PY" "$SCRIPT" --input "$SEL" --out "$OUT" --scratch "$SCRATCH" \
      --skip "$SKIP" --max "$CHUNK" --workers "$WORKERS"
echo "done task=$TASK rc=$?"
