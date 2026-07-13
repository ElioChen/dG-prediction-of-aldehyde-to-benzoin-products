#!/bin/bash
#SBATCH --job-name=confnoise
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/confnoise_chunks/cn_%A_%a.out
REPO="/scratch-shared/schen3/benzoin-dg"; D="$REPO/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
PY="/home/schen3/venv/nhc-workflow/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH" OMP_NUM_THREADS=1 XTB_BIN=/home/schen3/xtb/bin/xtb NCONF=5
W=24; PER=2; CH=$((W*PER)); ID=${SLURM_ARRAY_TASK_ID}; BASE=$((ID*CH))
for w in $(seq 0 $((W-1))); do
  SKIP=$((BASE + w*PER)); SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_${w}"; mkdir -p "$SCR"
  OUT="$D/confnoise_chunks/chunk_$(printf '%03d' $ID)_$(printf '%02d' $w).csv"
  [[ -s "$OUT" ]] && continue
  $PY -u "$REPO/pipeline/compute/conformer_noise_worker.py" --sample "$D/confnoise_sample.csv" \
      --skip $SKIP --max $PER --out "$OUT" --scratch "$SCR" &
done
wait; rm -rf /scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_* 2>/dev/null
echo "task $ID done $(date)"
