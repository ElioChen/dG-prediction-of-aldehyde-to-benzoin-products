#!/bin/bash
#SBATCH --job-name=cross_explore
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-93%40
#SBATCH --requeue
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/cross_explore_20260920/logs/explore_%A_%a.out
#
# 2026-09-20 (user: queue idle, small-scale exploratory expansion of cross
# data, ~750 uniform-random never-labeled pairs via FlyingDataset.sample()).
# Reuses rec1_b973c_tierB_worker.py as-is (same self-consistent
# conf_funnel_v3 + GFN2 --ohess + r2SCAN-3c/B97-3c/g-xTB SP recipe per
# species, product+donor+acceptor) -- genuinely new compute, unlike the
# homo success-rate fixes, since these products have never had a geometry
# computed before. Smoke-tested on 1 pair before submitting.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
SAMP="$REPO/data/cross_benzoin/cross_explore_20260920/cross_explore_pairs.csv"
OUTD="$REPO/data/cross_benzoin/cross_explore_20260920/shards"
CHUNK=8
mkdir -p "$OUTD" "$REPO/data/cross_benzoin/cross_explore_20260920/logs"
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
if [[ -f "$OUT.done" ]]; then echo "task $ID: $OUT.done exists, skip"; exit 0; fi

$PY -u "$REPO/cross_benzoin/rec1_b973c_tierB_worker.py" \
    --sample "$SAMP" --skip "$START" --nrows "$CHUNK" --out "$OUT" --scratch "$SCR" --xtb-cores 7
RC=$?
rm -rf "$SCR" 2>/dev/null
if [[ $RC -eq 0 ]]; then
  NROWS=$(( $(wc -l < "$OUT") - 1 ))
  EXPECT=$CHUNK
  TOT=$(( $(wc -l < "$SAMP") - 1 ))
  [[ $(( START + CHUNK )) -gt $TOT ]] && EXPECT=$(( TOT - START ))
  [[ "$NROWS" -ge "$EXPECT" ]] && touch "$OUT.done"
fi
echo "task $ID done rc=$RC rows=$(( $(wc -l < "$OUT") - 1 )) $(date)"
exit $RC
