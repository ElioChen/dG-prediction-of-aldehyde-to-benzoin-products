#!/bin/bash
#SBATCH --job-name=homo_sp
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-9202%250
#SBATCH --requeue
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/homo_standalone/relabel_sp/logs/hsp_%A_%a.out
#
# FAST homo relabel: DFT SP (r2SCAN-3c + B97-3c) on ARCHIVED geometries,
# reuse stored xTB thermal. 184,052 pairs / CHUNK 20 = 9,203 array tasks.
# Multi-arm: override --partition / --array on the CLI (Tier B style).
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
MAN="$REPO/data/cross_benzoin/homo_standalone/relabel_sp/homo_sp_manifest.csv"
OUTD="$REPO/data/cross_benzoin/homo_standalone/relabel_sp/shards"
CHUNK=20
mkdir -p "$OUTD" "$REPO/data/cross_benzoin/homo_standalone/relabel_sp/logs"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ORCA_BIN=/home/schen3/orca/orca ORCA_SCF=default
cd "$REPO"

ID=${SLURM_ARRAY_TASK_ID:-0}
START=$(( ID * CHUNK ))
if [[ -d /scratch-local ]]; then SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}"; else SCR="/tmp/${USER}.${SLURM_JOB_ID}_${ID}"; fi
mkdir -p "$SCR"
OUT="$OUTD/shard_$(printf '%05d' "$ID").csv"
if [[ -f "$OUT.done" ]]; then echo "task $ID: done, skip"; exit 0; fi

$PY -u "$REPO/cross_benzoin/homo_sp_from_geom_worker.py" \
    --manifest "$MAN" --skip "$START" --nrows "$CHUNK" --out "$OUT" \
    --scratch "$SCR" --methods "r2SCAN-3c,B97-3c" --sp-workers 16
RC=$?
rm -rf "$SCR" 2>/dev/null
if [[ $RC -eq 0 ]]; then
  NROWS=$(( $(wc -l < "$OUT") - 1 ))
  EXPECT=$CHUNK
  TOT=$(( $(wc -l < "$MAN") - 1 ))
  [[ $(( START + CHUNK )) -gt $TOT ]] && EXPECT=$(( TOT - START ))
  [[ "$NROWS" -ge "$EXPECT" ]] && touch "$OUT.done"
fi
echo "task $ID rc=$RC rows=$(( $(wc -l < "$OUT") - 1 )) $(date)"
exit $RC
