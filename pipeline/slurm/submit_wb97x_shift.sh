#!/bin/bash
#SBATCH --job-name=wb97x_shift
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/wb97x_shift/chunks/wb_%A_%a.out
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
SAMP="$REPO/data/cross_benzoin/wb97x_shift/sample.csv"
GEOMD="$REPO/data/cross_benzoin/wb97x_shift/geoms"
OUTD="$REPO/data/cross_benzoin/wb97x_shift/chunks"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH" OMP_NUM_THREADS=1
cd "$REPO"
W=7; PER=1; ID=${SLURM_ARRAY_TASK_ID:-0}; BASE=$((ID*W*PER))
for w in $(seq 0 $((W-1))); do
  SKIP=$((BASE + w*PER))
  if [[ -d /scratch-local ]]; then SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_${w}"; else SCR="/tmp/${USER}.${SLURM_JOB_ID}_${ID}_${w}"; fi
  mkdir -p "$SCR"
  OUT="$OUTD/wb_$(printf '%03d' "$ID")_$(printf '%02d' "$w").csv"
  [[ -s "$OUT" ]] && continue
  $PY -u "$REPO/cross_benzoin/wb97x_shift_worker.py" --sample "$SAMP" --geom-dir "$GEOMD" \
      --skip "$SKIP" --max "$PER" --out "$OUT" --scratch "$SCR" &
done
wait
rm -rf /scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_* /tmp/${USER}.${SLURM_JOB_ID}_${ID}_* 2>/dev/null
echo "task $ID done $(date)"
