#!/bin/bash
#SBATCH --job-name=cb10_fat_feat
#SBATCH --partition=fat_genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=18:00:00

set -euo pipefail

REPO="/gpfs/scratch1/shared/schen3/benzoin-dg-restored"
PKG="$REPO/cross_benzoin"
INPUT="${INPUT:?set INPUT=/abs/pairs.csv}"
OUTDIR="${OUTDIR:-$REPO/data/cross_benzoin/cross_round10}"
CHUNK="${CHUNK:-30}"

VENV="${VENV:-/home/schen3/venv/nhc-workflow}"
XTB_BIN="${XTB_BIN:-/home/schen3/xtb/bin/xtb}"
MWF_BIN="${MWF_BIN:-/home/schen3/mutiwfn/Multiwfn_noGUI}"
SOLVENT="${SOLVENT:-dmso}"
N_CONFS="${N_CONFS:-10}"
CONFORMER="${CONFORMER:-funnel_v3}"
MULTIWFN="${MULTIWFN:-0}"
EMIT_ALD="${EMIT_ALD:-1}"
WORKERS="${WORKERS:-12}"
XTB_CORES="${XTB_CORES:-2}"
ALD_CACHE="${ALD_CACHE:-}"
REQUIRE_CACHE_COMPLETE="${REQUIRE_CACHE_COMPLETE:-0}"

ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
TAG=$(printf "chunk_%04d" "$ID")
TASK_OUT="$OUTDIR/$TAG"
mkdir -p "$TASK_OUT" "$OUTDIR/logs"
DRIVER_LOG="$OUTDIR/logs/${TAG}.driver.log"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "cb10_fat_feat driver start ${SLURM_ARRAY_JOB_ID:-nojob}[$ID] $TAG node=${SLURMD_NODENAME:-unknown} $(date)"
echo "repo=$REPO input=$INPUT outdir=$OUTDIR task_out=$TASK_OUT"
[[ -f "$INPUT" ]] || { echo "ERROR: INPUT $INPUT not found"; exit 1; }

echo "checking explicit runtime paths"
# module loading intentionally skipped; all runtime paths are explicit
[[ -r "$VENV/bin/activate" ]] || { echo "ERROR: VENV activate not readable: $VENV/bin/activate"; exit 2; }
[[ -x "$VENV/bin/python" ]] || { echo "ERROR: VENV python not executable: $VENV/bin/python"; exit 2; }
[[ -x "$XTB_BIN" ]] || { echo "ERROR: XTB_BIN not executable: $XTB_BIN"; exit 2; }
export PATH="$VENV/bin:$PATH"

export XTBPATH="/home/schen3/xtb/share/xtb"
export GXTB_BIN="${GXTB_BIN:-$XTB_BIN}"
[[ -x "$GXTB_BIN" ]] || { echo "ERROR: GXTB_BIN not executable: $GXTB_BIN"; exit 2; }
export GXTB_SOLV="${GXTB_SOLV:-cosmo dmso}"
export OMP_NUM_THREADS="$XTB_CORES"
export MKL_NUM_THREADS="$XTB_CORES"
export OMP_STACKSIZE=2G
export KMP_STACKSIZE=2G

PAIRS_CSV="$TASK_OUT/pairs.csv"
"$VENV/bin/python" - "$INPUT" "$ID" "$CHUNK" "$PAIRS_CSV" <<'PY'
import sys
import pandas as pd

inp, i, chunk, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
df = pd.read_csv(inp)
lo, hi = i * chunk, min((i + 1) * chunk, len(df))
(df.iloc[lo:hi] if lo < len(df) else df.iloc[0:0]).to_csv(out, index=False)
print(f"chunk {i}: pairs {lo}:{hi}")
PY

N_EXPECT=$(($(wc -l < "$PAIRS_CSV") - 1))
N_DONE=0
[[ -f "$TASK_OUT/products.csv" ]] && N_DONE=$(($(wc -l < "$TASK_OUT/products.csv") - 1))
if [[ "$N_EXPECT" -gt 0 && "$N_DONE" -ge "$N_EXPECT" ]]; then
    echo "chunk $ID already done ($TAG): $N_DONE/$N_EXPECT rows - skip"
    exit 0
fi
[[ "$N_DONE" -gt 0 ]] && echo "chunk $ID partial output exists: $N_DONE/$N_EXPECT rows; recomputing chunk"

timeout 30 bash "$REPO/pipeline/slurm/clean_node_orphans.sh" 2>/dev/null || true

LOCAL_BASE="/scratch-local/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"
if [[ -d /scratch-local ]]; then
    export TMPDIR="$LOCAL_BASE"
else
    export TMPDIR="/tmp/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"
fi
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT TERM INT

MWF_ARGS=""
[[ "$MULTIWFN" == "1" ]] && MWF_ARGS="--multiwfn --multiwfn-bin $MWF_BIN"
ALD_ARGS=""
[[ "$EMIT_ALD" == "1" ]] && ALD_ARGS="--emit-aldehydes"
CACHE_ARGS=""
[[ -n "$ALD_CACHE" ]] && CACHE_ARGS="--aldehyde-cache $ALD_CACHE"
[[ "$REQUIRE_CACHE_COMPLETE" == "1" ]] && CACHE_ARGS="$CACHE_ARGS --require-cache-complete"

echo "cb10_fat_feat ${SLURM_ARRAY_JOB_ID}[$ID] $TAG node=${SLURMD_NODENAME:-unknown} chunk=$CHUNK workers=$WORKERS xtb_cores=$XTB_CORES cache=${ALD_CACHE:-none} require_cache=$REQUIRE_CACHE_COMPLETE tmp=$TMPDIR $(date)"
echo "outputs: $TASK_OUT/{pairs.csv,aldehydes.csv,products.csv,xyz_ald,xyz_prod,run.log}; transient files stay under TMPDIR and are removed on exit"
cd "$PKG"
set +e
"$VENV/bin/python" cb_featurize.py \
    --pairs "$PAIRS_CSV" --out "$TASK_OUT" \
    --xtb-bin "$XTB_BIN" --solvent "$SOLVENT" --n-confs "$N_CONFS" \
    --conformer "$CONFORMER" --workers "$WORKERS" --xtb-cores "$XTB_CORES" --parallel-jobs 1 \
    $MWF_ARGS $ALD_ARGS $CACHE_ARGS \
    2>&1 | tee "$TASK_OUT/run.log"
RC=${PIPESTATUS[0]}
set -e

echo "Done $TAG $(date) exit=$RC"
exit "$RC"
