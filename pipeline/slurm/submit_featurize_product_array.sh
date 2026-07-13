#!/bin/bash
#SBATCH --job-name=feat_product
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#
# Cross/Homo-Benzoin PRODUCT featurizer as a CHUNK-based SLURM array: ONE task = a
# CHUNK of aldehyde PAIRS. Each pair -> benzoin product, funnel_v3 conformer search
# (topology-guarded, SAVES product xyz), xTB+morfeus+Multiwfn descriptors + xTB ΔG.
# genoa bills in 24-core units, so cpus-per-task=24 (WORKERS*XTB_CORES = 24).
#
# Submit (create the log dir FIRST; INPUT = pairs CSV with donor_smiles,acceptor_smiles):
#   IN=/scratch-shared/schen3/benzoin-dg/data/raw/cross_pilot/pairs.csv
#   OUT=/scratch-shared/schen3/benzoin-dg/data/raw/cross_pilot
#   N=$(($(wc -l < "$IN")-1)); CHUNK=100; NCH=$(( (N+CHUNK-1)/CHUNK )); mkdir -p "$OUT/logs"
#   sbatch --array=0-$((NCH-1))%64 --output="$OUT/logs/prod_%A_%a.out" \
#     --export=ALL,INPUT="$IN",OUTDIR="$OUT",CHUNK=$CHUNK submit_featurize_product_array.sh
#
# Then concatenate per-chunk CSVs (identical headers); product xyz live in chunk_*/xyz/.
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
SCRIPT="$REPO/pipeline/compute/featurize_product.py"
INPUT="${INPUT:?set INPUT=/abs/pairs.csv}"
OUTDIR="${OUTDIR:-$REPO/data/raw/products}"
CHUNK="${CHUNK:-100}"

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"
MWF_BIN="/home/schen3/mutiwfn/Multiwfn_noGUI"
SOLVENT="${SOLVENT:-dmso}"; N_CONFS="${N_CONFS:-10}"
CONFORMER="${CONFORMER:-funnel_v3}"
MULTIWFN="${MULTIWFN:-1}"        # 1 = run Multiwfn (ADCH/QTAIM); 0 = xTB+morfeus only
EMIT_ALD="${EMIT_ALD:-1}"        # 1 = also featurize+save each aldehyde on its funnel_v3 geom
                                 #     (-> aldehydes.csv + ald_xyz/, method-consistent w/ product)
# 24 cores = WORKERS molecules in parallel x XTB_CORES threads each (parallel-jobs=1).
WORKERS="${WORKERS:-12}"; XTB_CORES="${XTB_CORES:-2}"

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

# Slice this task's chunk of pairs into its own CSV.
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

WORK_DIR="${TMPDIR:-$PROJ/tmp}/product_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT TERM INT

MWF_ARGS=""; [[ "$MULTIWFN" == "1" ]] && MWF_ARGS="--multiwfn --multiwfn-bin $MWF_BIN"
ALD_ARGS=""; [[ "$EMIT_ALD" == "1" ]] && ALD_ARGS="--emit-aldehydes --ald-output $TASK_OUT/aldehydes.csv"

echo "Product task ${SLURM_ARRAY_JOB_ID}[$ID] node=${SLURMD_NODENAME} chunk=$CHUNK conformer=$CONFORMER mwf=$MULTIWFN emit_ald=$EMIT_ALD $(date)"
cd "$REPO/pipeline/compute"
python "$SCRIPT" \
    --input "$PAIRS_CSV" --output "$TASK_OUT/features.csv" --work-dir "$WORK_DIR" \
    --xtb-bin "$XTB_BIN" --solvent "$SOLVENT" --n-confs "$N_CONFS" \
    --conformer "$CONFORMER" --workers "$WORKERS" --xtb-cores "$XTB_CORES" --parallel-jobs 1 \
    $MWF_ARGS $ALD_ARGS \
    2>&1 | tee "$TASK_OUT/run.log"
echo "Done $TAG $(date) exit=${PIPESTATUS[0]}"
