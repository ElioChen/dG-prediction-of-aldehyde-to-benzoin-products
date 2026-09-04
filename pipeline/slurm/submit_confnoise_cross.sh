#!/bin/bash
#SBATCH --job-name=confnoise_cross
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/confnoise_cross/chunks/cn_%A_%a.out
#
# cross-benzoin single-conformer LABEL-NOISE floor: K=5 conformers x r2SCAN-3c SP per
# product, per-mol DFT-energy std = ΔG spread. Analogue of the homo confnoise study
# (confnoise_summary_20260626_1559.md, ~2.14 kcal MAE floor). Confirms whether the
# r1-10 blend (MAE 2.215) is at the cross label-noise ceiling.
#   sbatch --array=0-3 pipeline/slurm/submit_confnoise_cross.sh   # 32 mols, 8/task
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
SAMP="$REPO/data/cross_benzoin/confnoise_sample_cross.csv"
OUTD="$REPO/data/cross_benzoin/confnoise_cross/chunks"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH" OMP_NUM_THREADS=1 XTB_BIN=/home/schen3/xtb/bin/xtb NCONF=5
cd "$REPO"
W=8; PER=1; CH=$((W*PER)); ID=${SLURM_ARRAY_TASK_ID:-0}; BASE=$((ID*CH))
for w in $(seq 0 $((W-1))); do
  SKIP=$((BASE + w*PER))
  if [[ -d /scratch-local ]]; then SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_${w}"; else SCR="/tmp/${USER}.${SLURM_JOB_ID}_${ID}_${w}"; fi
  mkdir -p "$SCR"
  OUT="$OUTD/chunk_$(printf '%03d' "$ID")_$(printf '%02d' "$w").csv"
  [[ -s "$OUT" ]] && continue
  $PY -u "$REPO/pipeline/compute/conformer_noise_worker.py" --sample "$SAMP" \
      --skip "$SKIP" --max "$PER" --out "$OUT" --scratch "$SCR" &
done
wait
rm -rf /scratch-local/${USER}.${SLURM_JOB_ID}_${ID}_* /tmp/${USER}.${SLURM_JOB_ID}_${ID}_* 2>/dev/null
echo "task $ID done $(date)"
