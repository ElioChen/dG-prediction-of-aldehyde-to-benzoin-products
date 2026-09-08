#!/bin/bash
#SBATCH --job-name=rec1_b973c_tierB
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-4440%250
#SBATCH --requeue
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/rec1_b973c_tierB/logs/tierB_%A_%a.out
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
SAMP="$REPO/data/cross_benzoin/rec1_b973c_tierB/rec1_b973c_tierB_pairs_35528.csv"
OUTD="$REPO/data/cross_benzoin/rec1_b973c_tierB/shards"
CHUNK=8
mkdir -p "$OUTD" "$REPO/data/cross_benzoin/rec1_b973c_tierB/logs"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XTB_BIN=/home/schen3/xtb/bin/xtb XTBPATH=/home/schen3/xtb/share/xtb
export ORCA_BIN=/home/schen3/orca/orca
export ORCA_SCF=default
cd "$REPO"

ID=${SLURM_ARRAY_TASK_ID:-0}
START=$(( ID * CHUNK ))
if [[ -d /scratch-local ]]; then SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}"; else SCR="/tmp/${USER}.${SLURM_JOB_ID}_${ID}"; fi
mkdir -p "$SCR"
OUT="$OUTD/shard_$(printf '%05d' "$ID").csv"

# resume-safe: worker skips pids already present in $OUT. Only a fully-complete shard
# (a .done marker) is skipped outright.
if [[ -f "$OUT.done" ]]; then echo "task $ID: $OUT.done exists, skip"; exit 0; fi

$PY -u "$REPO/cross_benzoin/rec1_b973c_tierB_worker.py" \
    --sample "$SAMP" --skip "$START" --nrows "$CHUNK" --out "$OUT" --scratch "$SCR" --xtb-cores 7
RC=$?
rm -rf "$SCR" 2>/dev/null
# mark shard done only if the worker returned 0 AND wrote CHUNK data rows (or fewer for the tail)
if [[ $RC -eq 0 ]]; then
  NROWS=$(( $(wc -l < "$OUT") - 1 ))
  EXPECT=$CHUNK
  TOT=$(( $(wc -l < "$SAMP") - 1 ))
  [[ $(( START + CHUNK )) -gt $TOT ]] && EXPECT=$(( TOT - START ))
  if [[ "$NROWS" -ge "$EXPECT" ]]; then touch "$OUT.done"; fi
fi
echo "task $ID done rc=$RC rows=$(( $(wc -l < "$OUT") - 1 )) $(date)"
exit $RC
