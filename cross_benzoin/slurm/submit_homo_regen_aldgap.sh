#!/bin/bash
#SBATCH --job-name=homo_regen_aldgap
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-1130%40
#SBATCH --requeue
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/homo_standalone/relabel_sp/regen_logs/hraldgap_%A_%a.out
#
# 2026-09-20 (user: queue idle, improve homo relabel success rate): 6,784
# pairs whose PRODUCT geometry extracted fine from the archive but whose
# ALDEHYDE geometry did not (geom_extract_fail(p=True,a=False) -- a missing
# xyz member in the tar, not a computability problem, since donor==acceptor
# and the product side proves the chemistry is fine). Distinct from the
# original ~5-10%-chunk-range geometry gap that homo_sp_manifest_regen.csv
# already covers (0 overlap, checked) -- this is a separate, narrower gap.
#
# rec_homo_relabel_worker.py doesn't support "only redo one species", so
# this regenerates BOTH product and aldehyde from SMILES via conf_funnel_v3
# + GFN2 --ohess (same recipe/cost as the original regen track, ~2-3 CPU-h/
# pair) rather than engineering a species-selective variant -- the redundant
# product recompute is an acceptable cost for reusing proven, already-tested
# infra with zero new code risk. Should push homo coverage 92.1% -> ~95.8%.
#
# Output goes into regen_shards/ with a distinguishing shard_aldgap_NNNNN.csv
# name -- merge_homo_sp.py's default glob (shard_*.csv) already picks it up.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
SAMP="$REPO/data/cross_benzoin/homo_standalone/relabel_sp/homo_sp_regen_pairs_aldgap.csv"
OUTD="$REPO/data/cross_benzoin/homo_standalone/relabel_sp/regen_shards"
CHUNK=6
mkdir -p "$OUTD" "$REPO/data/cross_benzoin/homo_standalone/relabel_sp/regen_logs"
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
OUT="$OUTD/shard_aldgap_$(printf '%05d' "$ID").csv"
if [[ -f "$OUT.done" ]]; then echo "task $ID done, skip"; exit 0; fi
$PY -u "$REPO/cross_benzoin/rec_homo_relabel_worker.py" \
    --sample "$SAMP" --skip "$START" --nrows "$CHUNK" --out "$OUT" --scratch "$SCR" --xtb-cores 7
RC=$?
rm -rf "$SCR" 2>/dev/null
if [[ $RC -eq 0 ]]; then
  NROWS=$(( $(wc -l < "$OUT") - 1 )); EXPECT=$CHUNK
  TOT=$(( $(wc -l < "$SAMP") - 1 ))
  [[ $(( START + CHUNK )) -gt $TOT ]] && EXPECT=$(( TOT - START ))
  [[ "$NROWS" -ge "$EXPECT" ]] && touch "$OUT.done"
fi
echo "task $ID rc=$RC rows=$(( $(wc -l < "$OUT") - 1 )) $(date)"
exit $RC
