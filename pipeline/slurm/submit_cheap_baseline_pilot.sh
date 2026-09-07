#!/bin/bash
#SBATCH --job-name=cheap_baseline_pilot
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --array=0-15%16
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/cheap_baseline_pilot/chunks/cbp_%A_%a.out
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
SAMP="$REPO/data/cross_benzoin/cheap_baseline_pilot/sample.csv"
OUTD="$REPO/data/cross_benzoin/cheap_baseline_pilot/chunks"
mkdir -p "$OUTD"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XTB_BIN=/home/schen3/xtb/bin/xtb XTBPATH=/home/schen3/xtb/share/xtb
cd "$REPO"
# 128 pairs, 8 workers/task, 16 tasks
W=8; PER=1; ID=${SLURM_ARRAY_TASK_ID:-0}; BASE=$((ID*W*PER))
for w in $(seq 0 $((W-1))); do
  SKIP=$((BASE + w*PER))
  if [[ -d /scratch-local ]]; then SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_${w}"; else SCR="/tmp/${USER}.${SLURM_JOB_ID}_${ID}_${w}"; fi
  mkdir -p "$SCR"
  OUT="$OUTD/cbp_$(printf '%03d' "$ID")_$(printf '%02d' "$w").csv"
  [[ -s "$OUT" ]] && continue
  $PY -u "$REPO/cross_benzoin/cheap_baseline_pilot_worker.py" --sample "$SAMP" \
      --skip "$SKIP" --max "$PER" --out "$OUT" --scratch "$SCR" &
done
wait
rm -rf /scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_* /tmp/${USER}.${SLURM_JOB_ID}_${ID}_* 2>/dev/null
echo "task $ID done $(date)"
