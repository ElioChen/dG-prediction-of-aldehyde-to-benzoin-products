#!/bin/bash
#SBATCH --job-name=cb_feat
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --time=12:00:00
#
# cross_benzoin unified featurizer as a CHUNK-based array. ONE task = a chunk of
# PAIRS -> cb_featurize.py (funnel_v3, Multiwfn, --emit-aldehydes). Per chunk writes
# chunk_XXXX/{aldehydes.csv,products.csv,xyz_ald/,xyz_prod/}. genoa 24-core unit.
#
# Submit (PAIRS csv: donor_id,acceptor_id,donor_smiles,acceptor_smiles):
#   IN=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/homo_pairs.csv
#   OUT=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6
#   N=$(($(wc -l < "$IN")-1)); CHUNK=100; NCH=$(( (N+CHUNK-1)/CHUNK )); mkdir -p "$OUT/logs"
#   sbatch --array=0-$((NCH-1))%64 --output="$OUT/logs/cb_%A_%a.out" \
#     --export=ALL,INPUT="$IN",OUTDIR="$OUT",CHUNK=$CHUNK submit_cb_featurize_array.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PKG="$REPO/cross_benzoin"
INPUT="${INPUT:?set INPUT=/abs/pairs.csv}"
OUTDIR="${OUTDIR:-$REPO/data/cross_benzoin/run}"
CHUNK="${CHUNK:-100}"

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"
MWF_BIN="/home/schen3/mutiwfn/Multiwfn_noGUI"
SOLVENT="${SOLVENT:-dmso}"; N_CONFS="${N_CONFS:-10}"
CONFORMER="${CONFORMER:-funnel_v3}"
# MULTIWFN defaults OFF: ADCH/QTAIM (Multiwfn L3) is ~minutes/molecule -> infeasible at
# 220k (the full-library adch_/qtaim_ columns are empty anyway, only filled on subsets),
# and the validated Δ-model path is Multiwfn-free. Keeping it off also slashes the per-
# molecule inode footprint (no molden/wfn/adch/qtaim files). Set MULTIWFN=1 only for a
# small subset run. EMIT_ALD stays on: the product side (xyz + xTB/g-xTB G) is REQUIRED
# for ΔG = G(prod) - 2·G(ald), independent of whether product descriptors help the ML.
MULTIWFN="${MULTIWFN:-0}"; EMIT_ALD="${EMIT_ALD:-1}"
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
# g-xTB baseline is fused into cb_featurize (one SP on the GFN2-ohess geom). Same env as
# submit_gxtb_baseline.sh; cb_featurize._gxtb_sp reads GXTB_BIN/GXTB_SOLV.
export GXTB_BIN="${GXTB_BIN:-/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb}"
export GXTB_SOLV="${GXTB_SOLV:-cosmo dmso}"
export OMP_NUM_THREADS=$XTB_CORES MKL_NUM_THREADS=$XTB_CORES OMP_STACKSIZE=2G KMP_STACKSIZE=2G

# Slice this task's chunk of pairs (always -- cheap, deterministic, idempotent -- so the
# resume check below can compare against the TRUE expected row count for this chunk).
PAIRS_CSV="$TASK_OUT/pairs.csv"
python - "$INPUT" "$ID" "$CHUNK" "$PAIRS_CSV" <<'PY'
import sys, pandas as pd
inp, i, chunk, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
df = pd.read_csv(inp)
lo, hi = i*chunk, min((i+1)*chunk, len(df))
(df.iloc[lo:hi] if lo < len(df) else df.iloc[0:0]).to_csv(out, index=False)
print(f"chunk {i}: pairs {lo}:{hi}")
PY

# Resume-safe: only skip if products.csv has a row for EVERY pair in this chunk. A
# products.csv that is merely non-empty is NOT sufficient -- cb_featurize.py flushes each
# row as it completes, so a task that is preempted, requeued, or hits the 12h wall clock
# mid-chunk leaves a partial, non-empty products.csv. The old (`-s`, non-empty) check
# would have silently treated that as fully done on any resubmit, permanently losing the
# un-run pairs -- plausibly the source of some of the ~0.15% (333/220,859) of aldehydes
# missing from aldehydes_all.csv after the original 220k campaign's multi-day, multi-
# requeue run.
N_EXPECT=$(($(wc -l < "$PAIRS_CSV") - 1))
N_DONE=0
[[ -f "$TASK_OUT/products.csv" ]] && N_DONE=$(($(wc -l < "$TASK_OUT/products.csv") - 1))
if [[ "$N_EXPECT" -gt 0 && "$N_DONE" -ge "$N_EXPECT" ]]; then
    echo "chunk $ID already done ($TAG): $N_DONE/$N_EXPECT rows — skip"
    exit 0
fi
[[ "$N_DONE" -gt 0 ]] && echo "chunk $ID resuming ($TAG): $N_DONE/$N_EXPECT rows previously written will be RECOMPUTED (cb_featurize.py has no row-level resume yet)"

# self-heal: clear this node's dead-job orphan scratch before generating our own. The
# per-user inode quota on the shared nodespecific tree is the bottleneck at %64/%128;
# per-id scontrol gate inside makes it safe for live jobs. timeout so a slow controller
# can't block job start.
timeout 30 bash "$REPO/pipeline/slurm/clean_node_orphans.sh" 2>/dev/null || true
export TMPDIR="${TMPDIR:-$PROJ/tmp}/cb_${SLURM_ARRAY_JOB_ID}_${ID}"; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT TERM INT

MWF_ARGS=""; [[ "$MULTIWFN" == "1" ]] && MWF_ARGS="--multiwfn --multiwfn-bin $MWF_BIN"
ALD_ARGS=""; [[ "$EMIT_ALD" == "1" ]] && ALD_ARGS="--emit-aldehydes"

echo "cb_feat ${SLURM_ARRAY_JOB_ID}[$ID] $TAG node=${SLURMD_NODENAME} conformer=$CONFORMER mwf=$MULTIWFN emit_ald=$EMIT_ALD $(date)"
cd "$PKG"
python cb_featurize.py \
    --pairs "$PAIRS_CSV" --out "$TASK_OUT" \
    --xtb-bin "$XTB_BIN" --solvent "$SOLVENT" --n-confs "$N_CONFS" \
    --conformer "$CONFORMER" --workers "$WORKERS" --xtb-cores "$XTB_CORES" --parallel-jobs 1 \
    $MWF_ARGS $ALD_ARGS \
    2>&1 | tee "$TASK_OUT/run.log"
echo "Done $TAG $(date) exit=${PIPESTATUS[0]}"
