#!/bin/bash
#SBATCH --job-name=homo_relabel
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-13469%250
#SBATCH --requeue
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/homo_standalone/relabel/logs/hr_%A_%a.out
#
# HOMO full-library self-consistent B97-3c relabel campaign.
# 161,630 pairs (2,000 front QC + 159,630 relabel) / CHUNK 12 = 13,470 array tasks.
# Homo worker does 2 species/pair (prod + aldehyde), dG = G_prod - 2*G_ald.
#
# Multi-arm: override --partition and --array on the sbatch CLI, e.g.
#   sbatch --partition=fat_genoa --array=6735-13469%150 cross_benzoin/slurm/submit_homo_relabel.sh
# rome/genoa QOS MaxJobsPU=128; fat_rome/fat_genoa uncapped but +50% billed.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
SAMP="$REPO/data/cross_benzoin/homo_standalone/homo_relabel_pairs.csv"
OUTD="$REPO/data/cross_benzoin/homo_standalone/relabel/shards"
CHUNK=12
mkdir -p "$OUTD" "$REPO/data/cross_benzoin/homo_standalone/relabel/logs"
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

$PY -u "$REPO/cross_benzoin/rec_homo_relabel_worker.py" \
    --sample "$SAMP" --skip "$START" --nrows "$CHUNK" --out "$OUT" --scratch "$SCR" --xtb-cores 7
RC=$?
rm -rf "$SCR" 2>/dev/null
if [[ $RC -eq 0 ]]; then
  NROWS=$(( $(wc -l < "$OUT") - 1 ))
  EXPECT=$CHUNK
  TOT=$(( $(wc -l < "$SAMP") - 1 ))
  [[ $(( START + CHUNK )) -gt $TOT ]] && EXPECT=$(( TOT - START ))
  if [[ "$NROWS" -ge "$EXPECT" ]]; then touch "$OUT.done"; fi
fi
echo "task $ID done rc=$RC rows=$(( $(wc -l < "$OUT") - 1 )) $(date)"
exit $RC
