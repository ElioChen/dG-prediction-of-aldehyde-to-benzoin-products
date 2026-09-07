#!/bin/bash
#SBATCH --job-name=rec1_prodgeom
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --array=0-29%12
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/rec1_prodgeom_recheck/chunks/rec1_%A_%a.out
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
SAMP="$REPO/data/cross_benzoin/rec1_prodgeom_recheck/rec1_recheck_pairs30.csv"
OUTD="$REPO/data/cross_benzoin/rec1_prodgeom_recheck/chunks"
mkdir -p "$OUTD"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XTB_BIN=/home/schen3/xtb/bin/xtb XTBPATH=/home/schen3/xtb/share/xtb
export ORCA_BIN=/home/schen3/orca/orca
export ORCA_SCF=default        # r2SCAN-3c SP: ORCA default (normal) SCF, ~20-40% faster, <0.1 kcal on rel dG
cd "$REPO"

ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ -d /scratch-local ]]; then SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}"; else SCR="/tmp/${USER}.${SLURM_JOB_ID}_${ID}"; fi
mkdir -p "$SCR"
OUT="$OUTD/rec1_$(printf '%03d' "$ID").csv"
if [[ -s "$OUT" ]]; then echo "task $ID: $OUT exists, skip"; exit 0; fi

$PY -u "$REPO/cross_benzoin/rec1_prodgeom_recheck_worker.py" \
    --sample "$SAMP" --skip "$ID" --max 1 --out "$OUT" --scratch "$SCR" --xtb-cores 11
RC=$?
rm -rf "$SCR" 2>/dev/null
echo "task $ID done rc=$RC $(date)"
exit $RC
