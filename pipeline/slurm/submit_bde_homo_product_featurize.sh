#!/bin/bash
#SBATCH --job-name=bde_homoprod
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#
# Rebuild the BDE-project homo descriptor libraries destroyed by the 2026-07 Snellius
# purge: data/cross_benzoin/homo_v6/{products_all.csv, aldehydes_all.csv}. Both were
# gitignored (*_all.csv), so no GitHub branch has them; the 2026-09-02 recovery only
# recomputed 42k/220k aldehydes (round10 pairs + rounds 1-7) and NO products at all.
# See pipeline/bde/STATUS.md section 五 and memory bde-post-purge-asset-state.
#
# One array task = a CHUNK of homo pairs (donor == acceptor). --emit-aldehydes means the
# same run also writes method-consistent aldehyde descriptors, so this ONE product-phase
# campaign covers both libraries (the product phase is ~6x the aldehyde phase, so running
# aldehydes separately would be pure waste). Multiwfn ON: B6's product LOCAL_FEATURES
# needs the adch_*/qtaim_* columns.
#
# Submit (log dir is created here):
#   REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
#   OUT=$REPO/data/cross_benzoin/bde_homo_product_featurize_20260902
#   IN=$OUT/homo_pairs_all.csv
#   N=$(($(wc -l < "$IN")-1)); CHUNK=100; NCH=$(( (N+CHUNK-1)/CHUNK ))
#   sbatch --array=0-$((NCH-1))%96 --output="$OUT/logs/prod_%A_%a.out" \
#     --export=ALL,REPO="$REPO",INPUT="$IN",OUTDIR="$OUT",CHUNK=$CHUNK \
#     pipeline/slurm/submit_bde_homo_product_featurize.sh
#
# Then: python pipeline/bde/assemble_homo_descriptor_libs.py   (concat + id-remap + merge)
#
set -eo pipefail   # NOT -u: `source /etc/profile` and venv activate reference unset vars

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
SCRIPT="$REPO/pipeline/compute/featurize_product.py"
INPUT="${INPUT:?set INPUT=/abs/homo_pairs.csv}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"
CHUNK="${CHUNK:-100}"
echo "bde_homoprod start $(date) host=$(hostname) REPO=$REPO INPUT=$INPUT CHUNK=$CHUNK"

VENV="${VENV:-/home/schen3/venv/nhc-workflow}"
XTB_BIN="${XTB_BIN:-/home/schen3/xtb/bin/xtb}"
MWF_BIN="${MWF_BIN:-/home/schen3/mutiwfn/Multiwfn_noGUI}"
SOLVENT="${SOLVENT:-dmso}"
N_CONFS="${N_CONFS:-10}"
CONFORMER="${CONFORMER:-funnel_v3}"
WORKERS="${WORKERS:-12}"
XTB_CORES="${XTB_CORES:-2}"

ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
TAG=$(printf "chunk_%04d" "$ID")
TASK_OUT="$OUTDIR/$TAG"
mkdir -p "$TASK_OUT" "$OUTDIR/logs"
[[ -f "$INPUT" ]] || { echo "ERROR: INPUT $INPUT not found"; exit 1; }
[[ -x "$XTB_BIN" ]] || { echo "ERROR: XTB_BIN not executable: $XTB_BIN"; exit 2; }
[[ -x "$MWF_BIN" ]] || { echo "ERROR: MWF_BIN not executable: $MWF_BIN"; exit 2; }

source /etc/profile 2>/dev/null || true
module load 2023 2>/dev/null || true
source "$VENV/bin/activate" || { echo "ERROR: venv activate failed: $VENV"; exit 3; }
export XTBPATH="/home/schen3/xtb/share/xtb"
export OMP_NUM_THREADS=$XTB_CORES MKL_NUM_THREADS=$XTB_CORES OMP_STACKSIZE=2G KMP_STACKSIZE=2G

# Slice this task's chunk of pairs.
PAIRS_CSV="$TASK_OUT/pairs.csv"
python - "$INPUT" "$ID" "$CHUNK" "$PAIRS_CSV" <<'PY'
import sys, pandas as pd
inp, i, chunk, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
df = pd.read_csv(inp)
lo, hi = i*chunk, min((i+1)*chunk, len(df))
if lo >= len(df):
    open(out, "w").write(",".join(df.columns)+"\n"); sys.exit(0)
df.iloc[lo:hi].to_csv(out, index=False)
print(f"chunk {i}: pairs {lo}:{hi} ({hi-lo})")
PY

# resume guard: a finished chunk's features.csv has >= (n_pairs) data rows.
N_PAIRS=$(($(wc -l < "$PAIRS_CSV") - 1))
N_DONE=0
[[ -f "$TASK_OUT/features.csv" ]] && N_DONE=$(($(wc -l < "$TASK_OUT/features.csv") - 1))
if [[ "$N_PAIRS" -gt 0 && "$N_DONE" -ge "$N_PAIRS" ]]; then
    echo "chunk $ID already done ($TAG): $N_DONE rows for $N_PAIRS pairs - skip"
    exit 0
fi

LOCAL_BASE="/scratch-local/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"
if [[ -d /scratch-local ]]; then export TMPDIR="$LOCAL_BASE"; else export TMPDIR="/tmp/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"; fi
WORK_DIR="$TMPDIR/work"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$TMPDIR"' EXIT TERM INT

echo "bde_homoprod ${SLURM_ARRAY_JOB_ID:-nojob}[$ID] $TAG node=${SLURMD_NODENAME:-?} chunk=$CHUNK $(date)"
cd "$REPO/pipeline/compute"
set +e
python "$SCRIPT" \
    --input "$PAIRS_CSV" --output "$TASK_OUT/features.csv" --work-dir "$WORK_DIR" \
    --xtb-bin "$XTB_BIN" --multiwfn --multiwfn-bin "$MWF_BIN" \
    --solvent "$SOLVENT" --n-confs "$N_CONFS" --conformer "$CONFORMER" \
    --workers "$WORKERS" --xtb-cores "$XTB_CORES" --parallel-jobs 1 \
    --emit-aldehydes --ald-output "$TASK_OUT/aldehydes.csv" \
    2>&1 | tee "$TASK_OUT/run.log"
RC=${PIPESTATUS[0]}
set -e
echo "Done $TAG $(date) exit=$RC"
exit "$RC"
