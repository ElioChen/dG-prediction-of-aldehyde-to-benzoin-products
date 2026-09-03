#!/bin/bash
#SBATCH --job-name=r89_ald_regen
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/r89_aldehyde_regen/logs/ald_%A_%a.out
#
# Regenerate funnel_v3 xTB geometries + thermal (G_xtb, xtb_energy) for the 20,617
# unique rounds 8-9 aldehydes -- the aldehyde leg of the r8/9 DFT-label recovery.
# Chosen over harvesting from Window A's 220k homo featurize because harvesting is
# slow (per-geom tar --zstd -xf over GPFS), fragile (empty aldehydes.csv mid-write
# crashes it), and gated on Window A finishing (~9/4). This is self-contained and
# emits geometry + xTB thermal in ONE pass. Same path as the r1-7 recovery
# (submit_cb_featurize_aldehyde_recovery.sh).
#
# Submit:
#   OUTDIR=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/r89_aldehyde_regen
#   IN=$OUTDIR/r89_aldehyde_todo.csv
#   N=$(( $(wc -l < "$IN") - 1 )); CHUNK=50; NT=$(( (N+CHUNK-1)/CHUNK ))
#   mkdir -p "$OUTDIR/logs"
#   sbatch --array=0-$((NT-1))%80 \
#     --export=ALL,INPUT="$IN",OUTDIR="$OUTDIR",CHUNK=$CHUNK \
#     cross_benzoin/slurm/submit_r89_aldehyde_regen.sh
#
set -euo pipefail
REPO="/gpfs/scratch1/shared/schen3/benzoin-dg-restored"
PKG="$REPO/cross_benzoin"
INPUT="${INPUT:?set INPUT=/abs/r89_aldehyde_todo.csv}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"
CHUNK="${CHUNK:-50}"
VENV="${VENV:-/home/schen3/venv/nhc-workflow}"
XTB_BIN="${XTB_BIN:-/home/schen3/xtb/bin/xtb}"
SOLVENT="${SOLVENT:-dmso}"; N_CONFS="${N_CONFS:-10}"; CONFORMER="${CONFORMER:-funnel_v3}"
WORKERS="${WORKERS:-12}"; XTB_CORES="${XTB_CORES:-2}"

ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
TAG=$(printf "chunk_%04d" "$ID")
TASK_OUT="$OUTDIR/$TAG"
mkdir -p "$TASK_OUT" "$OUTDIR/logs"
[[ -f "$INPUT" ]] || { echo "ERROR: INPUT $INPUT not found"; exit 1; }
[[ -x "$XTB_BIN" ]] || { echo "ERROR: XTB_BIN not executable: $XTB_BIN"; exit 2; }
export PATH="$VENV/bin:$PATH"
export XTBPATH="/home/schen3/xtb/share/xtb"
# cb_featurize.py hardcodes a PURGED g-xtb path as the GXTB_BIN default -- without this
# override every aldehyde row errors with "No such file: .../software/g-xtb/.../xtb".
# /home/schen3/xtb is g-xtb-capable (--gxtb). Mirrors submit_cb_featurize_aldehyde_recovery.sh.
export GXTB_BIN="${GXTB_BIN:-$XTB_BIN}"
export GXTB_SOLV="${GXTB_SOLV:-cosmo dmso}"
export OMP_NUM_THREADS="$XTB_CORES" MKL_NUM_THREADS="$XTB_CORES" OMP_STACKSIZE=2G KMP_STACKSIZE=2G

SLICE="$TASK_OUT/todo.csv"
"$VENV/bin/python" - "$INPUT" "$ID" "$CHUNK" "$SLICE" <<'PY'
import sys, pandas as pd
inp, i, chunk, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
df = pd.read_csv(inp)
lo, hi = i*chunk, min((i+1)*chunk, len(df))
(df.iloc[lo:hi] if lo < len(df) else df.iloc[0:0]).to_csv(out, index=False)
print(f"chunk {i}: rows {lo}:{hi}")
PY

# resume guard: a finished chunk has >= N_ROWS ERROR-FREE aldehyde rows. Counting
# raw rows (as an earlier version did) let a chunk full of errored rows from the
# cancelled broken-GXTB attempt-1 be trusted and skipped -> the 26351006 task 15-41
# hole (see RUN_LOG 2026-09-03 18:16). Count only rows whose `error` column is empty.
N_ROWS=$(($(wc -l < "$SLICE") - 1))
N_DONE=0
if [[ -f "$TASK_OUT/aldehydes.csv" ]]; then
  N_DONE=$("$VENV/bin/python" - "$TASK_OUT/aldehydes.csv" <<'PY'
import sys, pandas as pd
try:
    d = pd.read_csv(sys.argv[1])
    e = d["error"] if "error" in d.columns else pd.Series([None]*len(d))
    print(int((e.isna() | (e.astype(str).str.strip() == "")).sum()))
except Exception:
    print(0)
PY
)
fi
if [[ "$N_ROWS" -gt 0 && "$N_DONE" -ge "$N_ROWS" ]]; then
  echo "chunk $ID already done ($N_DONE/$N_ROWS error-free) - skip"; exit 0
fi

if [[ -d /scratch-local ]]; then export TMPDIR="/scratch-local/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"
else export TMPDIR="/tmp/${USER}.${SLURM_ARRAY_JOB_ID}_${ID}"; fi
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT TERM INT HUP QUIT

cd "$PKG"
set +e
"$VENV/bin/python" cb_featurize.py \
  --homo-from "$SLICE" --out "$TASK_OUT" --aldehydes-only --emit-aldehydes \
  --xtb-bin "$XTB_BIN" --solvent "$SOLVENT" --n-confs "$N_CONFS" \
  --conformer "$CONFORMER" --workers "$WORKERS" --xtb-cores "$XTB_CORES" --parallel-jobs 1 \
  2>&1 | tee "$TASK_OUT/run.log"
RC=${PIPESTATUS[0]}
set -e
echo "Done $TAG $(date) exit=$RC"
exit "$RC"
