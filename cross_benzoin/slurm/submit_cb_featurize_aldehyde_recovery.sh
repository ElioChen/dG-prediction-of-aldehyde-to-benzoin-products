#!/bin/bash
#SBATCH --job-name=ald_recover
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#
# Rebuild the aldehyde descriptor library that the 2026-07 Snellius purge destroyed
# (homo_v6/aldehydes_all.csv was gitignored, so no branch of the GitHub remote has it).
# Runs cb_featurize.py's ALDEHYDE PHASE ONLY (--aldehydes-only): every product this
# library is needed for already survives in the tracked cross_round*_dft_products.csv
# tables, and the product phase costs ~6x the aldehyde phase (stage1: 6 min vs 39 min
# per 20 molecules), so recomputing products here would be pure waste.
#
# Submit:
#   OUTDIR=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/aldehyde_recovery_r17
#   sbatch --array=0-910%60 --output="$OUTDIR/logs/ald_%a.out" \
#     --export=ALL,INPUT="$OUTDIR/recovery_pairs.csv",OUTDIR="$OUTDIR",CHUNK=15 \
#     cross_benzoin/slurm/submit_cb_featurize_aldehyde_recovery.sh
#
set -euo pipefail

REPO="/gpfs/scratch1/shared/schen3/benzoin-dg-restored"
PKG="$REPO/cross_benzoin"
INPUT="${INPUT:?set INPUT=/abs/pairs.csv}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"
CHUNK="${CHUNK:-15}"

VENV="${VENV:-/home/schen3/venv/nhc-workflow}"
XTB_BIN="${XTB_BIN:-/home/schen3/xtb/bin/xtb}"
SOLVENT="${SOLVENT:-dmso}"
N_CONFS="${N_CONFS:-10}"
CONFORMER="${CONFORMER:-funnel_v3}"
WORKERS="${WORKERS:-12}"
XTB_CORES="${XTB_CORES:-2}"

ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
TAG=$(printf "chunk_%04d" "$ID")
TASK_OUT="$OUTDIR/$TAG"
mkdir -p "$TASK_OUT" "$OUTDIR/logs"

echo "ald_recover ${SLURM_ARRAY_JOB_ID:-nojob}[$ID] $TAG node=${SLURMD_NODENAME:-unknown} $(date)"
[[ -f "$INPUT" ]] || { echo "ERROR: INPUT $INPUT not found"; exit 1; }
[[ -r "$VENV/bin/activate" ]] || { echo "ERROR: VENV not readable: $VENV"; exit 2; }
[[ -x "$XTB_BIN" ]] || { echo "ERROR: XTB_BIN not executable: $XTB_BIN"; exit 2; }
export PATH="$VENV/bin:$PATH"
export XTBPATH="/home/schen3/xtb/share/xtb"
export GXTB_BIN="${GXTB_BIN:-$XTB_BIN}"
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

# resume guard: a finished chunk emits one aldehydes.csv row per unique molecule in its
# pair slice (<= 2 * n_pairs, fewer when a molecule repeats), so require at least the
# pair count before declaring the chunk done.
N_PAIRS=$(($(wc -l < "$PAIRS_CSV") - 1))
N_DONE=0
[[ -f "$TASK_OUT/aldehydes.csv" ]] && N_DONE=$(($(wc -l < "$TASK_OUT/aldehydes.csv") - 1))
if [[ "$N_PAIRS" -gt 0 && "$N_DONE" -ge "$N_PAIRS" ]]; then
    echo "chunk $ID already done ($TAG): $N_DONE aldehyde rows for $N_PAIRS pairs - skip"
    exit 0
fi

timeout 30 bash "$REPO/pipeline/slurm/clean_node_orphans.sh" 2>/dev/null || true

LOCAL_BASE="/scratch-local/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"
if [[ -d /scratch-local ]]; then
    export TMPDIR="$LOCAL_BASE"
else
    export TMPDIR="/tmp/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"
fi
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT TERM INT

cd "$PKG"
set +e
"$VENV/bin/python" cb_featurize.py \
    --pairs "$PAIRS_CSV" --out "$TASK_OUT" --aldehydes-only --emit-aldehydes \
    --xtb-bin "$XTB_BIN" --solvent "$SOLVENT" --n-confs "$N_CONFS" \
    --conformer "$CONFORMER" --workers "$WORKERS" --xtb-cores "$XTB_CORES" --parallel-jobs 1 \
    2>&1 | tee "$TASK_OUT/run.log"
RC=${PIPESTATUS[0]}
set -e

echo "Done $TAG $(date) exit=$RC"
exit "$RC"
