#!/bin/bash
#SBATCH --job-name=boltzrelab_hard
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/boltz_chunks_hardtail_20260710/br_%A_%a.out
REPO="/scratch-shared/schen3/benzoin-dg"; D="$REPO/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
PY="/home/schen3/venv/nhc-workflow/bin/python"
mkdir -p "$D/boltz_chunks_hardtail_20260710"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH" OMP_NUM_THREADS=1 XTB_BIN=/home/schen3/xtb/bin/xtb NCONF=10
# hard-tail probe (2026-07-10): targeted top-150-|error| sulfonyl/P/imine/amide molecules,
# K=10 conformers (2x the original 20260626 pilot's K=5), scored against the CURRENT champion
W=24; PER=1; CH=$((W*PER)); ID=${SLURM_ARRAY_TASK_ID}; BASE=$((ID*CH))
for w in $(seq 0 $((W-1))); do
  SKIP=$((BASE + w*PER)); SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_${w}"; mkdir -p "$SCR"
  OUT="$D/boltz_chunks_hardtail_20260710/chunk_$(printf '%03d' $ID)_$(printf '%02d' $w).csv"
  [[ -s "$OUT" ]] && continue
  $PY -u "$REPO/pipeline/compute/boltz_relabel_worker.py" --sample "$D/boltz_relabel_hardtail_sample_20260710.csv" \
      --skip $SKIP --max $PER --out "$OUT" --scratch "$SCR" &
done
wait; rm -rf /scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_* 2>/dev/null
echo "task $ID done $(date)"
