#!/bin/bash
#SBATCH --job-name=dftsp_cross
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_cross/logs/dftsp_cross_%j.out
#
# Cross-benzoin r2SCAN-3c ΔG by SP on SAVED cross_pilot_v1 product geometries
# (pipeline/compute/dft_sp_cross_from_geom.py). The aldehyde (donor+acceptor) side needs
# ZERO new DFT compute -- it reuses the existing full-library homo DFT-SP campaign
# (data/raw/dft_sp_funnelv3, ~219k aldehydes already at r2SCAN-3c). Only the 598 NEW
# product geometries (cross substrate combinations never computed before) get a fresh SP.
#
# Small enough (598 rows) to run as ONE task, no array chunking needed. Smoke-tested
# 2026-07-14 on the login node (3/3 ok, sane sign/magnitude DFT corrections).
#
#   sbatch pipeline/slurm/submit_dft_sp_cross.sh
#
set -o pipefail
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
PRODUCTS="$REPO/data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_products.csv"
OUT="$REPO/data/raw/dft_sp_cross/cross_pilot_v1_dft_sp.csv"
WORKERS="${WORKERS:-48}"; MAXCORE="${MAXCORE:-1500}"; TIMEOUT="${TIMEOUT:-7200}"
ORCA_BIN="/home/schen3/orca/orca"
mkdir -p "$REPO/data/raw/dft_sp_cross/logs"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# No submit-script-level resume skip here: dft_sp_cross_from_geom.py now does its own
# row-level resume (skips ids already present in $OUT, appends+flushes incrementally),
# so a re-submit after a timeout/preemption is safe and cheap even if $OUT is non-empty
# but incomplete -- unlike the old all-at-once-write version this replaced.

echo "dftsp_cross workers=$WORKERS node=${SLURMD_NODENAME} $(date)"
$PY -u "$REPO/pipeline/compute/dft_sp_cross_from_geom.py" \
    --products-csv "$PRODUCTS" --skip 0 --max 1000 \
    --out-csv "$OUT" --workers "$WORKERS" --maxcore "$MAXCORE" \
    --timeout "$TIMEOUT" --orca-bin "$ORCA_BIN"
echo "Done $(date)"
