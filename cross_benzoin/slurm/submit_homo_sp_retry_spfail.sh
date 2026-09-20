#!/bin/bash
#SBATCH --job-name=homo_sp_retry
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-57%30
#SBATCH --requeue
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/homo_standalone/relabel_sp/logs/hspretry_%A_%a.out
#
# 2026-09-20 (user: queue idle, improve the relabel campaign's success rate):
# retry the 1,154 pairs that failed at the DFT-SP step (997 r2SCAN-3c:sp_fail
# + 157 bad_thermal) -- geometry extraction succeeded for these, so a plain
# retry of homo_sp_from_geom_worker.py (same manifest schema, same recipe)
# is the cheap thing to try before assuming they're hard failures. Output
# goes into the SAME shards/ dir merge_homo_sp.py already globs
# (shard_*.csv matches shard_retry_NNNNN.csv too) -- its own
# sort-by-success/drop_duplicates("id") logic means a successful retry here
# automatically wins over the old failed row on the next merge, no separate
# reconciliation step needed.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
MAN="$REPO/data/cross_benzoin/homo_standalone/relabel_sp/homo_sp_manifest_retry_spfail.csv"
OUTD="$REPO/data/cross_benzoin/homo_standalone/relabel_sp/shards"
CHUNK=20
mkdir -p "$OUTD" "$REPO/data/cross_benzoin/homo_standalone/relabel_sp/logs"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XTB_BIN=/home/schen3/xtb/bin/xtb XTBPATH=/home/schen3/xtb/share/xtb
export ORCA_BIN=/home/schen3/orca/orca ORCA_SCF=default
cd "$REPO"

ID=${SLURM_ARRAY_TASK_ID:-0}
START=$(( ID * CHUNK ))
if [[ -d /scratch-local ]]; then SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}"; else SCR="/tmp/${USER}.${SLURM_JOB_ID}_${ID}"; fi
mkdir -p "$SCR"
OUT="$OUTD/shard_retry_$(printf '%05d' "$ID").csv"
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
