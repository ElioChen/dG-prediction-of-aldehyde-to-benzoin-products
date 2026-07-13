#!/bin/bash
#SBATCH --job-name=mwf_adq
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/mwf_chunks/mwf_%A_%a.out
REPO="/scratch-shared/schen3/benzoin-dg"; D="$REPO/data/cross_benzoin/homo_v6/viz_gxtb_20260625"
PY="/home/schen3/venv/nhc-workflow/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:$PATH" OMP_NUM_THREADS=1 XTB_BIN=/home/schen3/xtb/bin/xtb MWF_BIN=/home/schen3/mutiwfn/Multiwfn_noGUI
W=24; PER=5; CH=$((W*PER)); ID=${SLURM_ARRAY_TASK_ID}; BASE=$((ID*CH))
for w in $(seq 0 $((W-1))); do
  SKIP=$((BASE + w*PER)); SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_${w}"; mkdir -p "$SCR"
  OUT="$D/mwf_chunks/chunk_$(printf '%05d' $ID)_$(printf '%02d' $w).csv"
  [[ -s "$OUT" ]] && continue
  $PY -u "$REPO/pipeline/compute/mwf_subset_worker.py" --subset "$D/adchqtaim_subset.csv" \
      --skip $SKIP --max $PER --out "$OUT" --scratch "$SCR" &
done
wait
rm -rf /scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_* 2>/dev/null
echo "task $ID done $(date)"
