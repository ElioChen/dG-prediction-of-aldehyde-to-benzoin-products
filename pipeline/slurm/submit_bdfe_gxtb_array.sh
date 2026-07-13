#!/bin/bash
#SBATCH --job-name=bdfe_gxtb_pilot
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#
# g-xTB-consistent BDFE pilot: scattered chunk-id sample across the full id range to get a
# representative (size/aromatic-vs-aliphatic diverse) timing + correctness read before
# considering a full-library array (same cost as the completed GFN2 v2 run, ~1 day, plus a
# modest g-xTB SP addendum -- see calc_bde_free_energy_gxtb.py docstring for why the
# fragment ohess must be redone rather than reused).
#
# Submit (WHICH=aldehydes|products):
#   WHICH=aldehydes; CHUNKSIZE=20
#   OUT=data/cross_benzoin/homo_v6/bdfe_gxtb_pilot_${WHICH}; mkdir -p "$OUT/logs"
#   sbatch --array=0,200,400,600,800,1000,1200 --output="$OUT/logs/bdfe_gxtb_%a.out" \
#     --export=ALL,WHICH=$WHICH,CHUNKSIZE=$CHUNKSIZE,OUTDIR=$OUT pipeline/slurm/submit_bdfe_gxtb_array.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
# Isolated lightweight env (numpy/pandas/rdkit/xgboost/scipy/sklearn/joblib/matplotlib only),
# NOT the shared envs/gnn -- that env got corrupted mid-run by an unrelated concurrent full
# environment rebuild (291 packages touched, rdkit/numpy/pandas all broken; see
# shared-env-instability-2026-07-05 memory). This env is dedicated to this pipeline so it
# can never again be disturbed by other concurrent work in envs/gnn.
PY="/gpfs/scratch1/shared/schen3/envs/bde_lite/bin/python"
XTB_BIN="/home/schen3/xtb/bin/xtb"
GXTB_BIN="/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb"
WHICH="${WHICH:?set WHICH=aldehydes|products}"
CHUNKSIZE="${CHUNKSIZE:-20}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"

ID=$SLURM_ARRAY_TASK_ID
echo "bdfe_gxtb ${SLURM_ARRAY_JOB_ID}[$ID] which=$WHICH node=${SLURMD_NODENAME} $(date)"
cd "$REPO/pipeline/compute"
WORK="${TMPDIR:-$REPO/tmp}/bdfe_gxtb_${SLURM_ARRAY_JOB_ID}_${ID}"
t0=$(date +%s)
# Under heavy concurrency, the shared conda env's rdkit import has been observed to fail
# transiently (GPFS read contention on the shared .so files -- caught via a 27.6% silent
# failure rate at %128 concurrency, see bde-descriptor-idea memory). Retry once after a
# short sleep before giving up, and propagate the REAL exit code so SLURM marks a genuine
# failure as FAILED instead of silently COMPLETED (the previous version had no exit-code
# check here at all).
rc=1
for attempt in 1 2; do
    $PY -u calc_bde_free_energy_gxtb.py --which "$WHICH" --chunk-id "$ID" --chunk-size "$CHUNKSIZE" \
        --out-dir "$OUTDIR" --xtb-bin "$XTB_BIN" --gxtb-bin "$GXTB_BIN" --work-dir "$WORK"
    rc=$?
    if [ $rc -eq 0 ]; then break; fi
    echo "attempt $attempt failed (exit=$rc), retrying after sleep..."
    sleep 10
done
t1=$(date +%s)
rm -rf "$WORK"
echo "Done task=$ID $(date) elapsed=$((t1-t0))s exit=$rc"
exit $rc
