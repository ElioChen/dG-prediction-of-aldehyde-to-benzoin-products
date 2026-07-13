#!/bin/bash
#SBATCH --job-name=dftopt_bench
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#
# DFT-geometry benchmark: r2SCAN-3c Opt vs r2SCAN-3c//xTB SP, to isolate the
# GEOMETRY term in the benzoin reaction ΔG (does DFT-opt help beyond DFT-SP?).
# One molecule (aldehyde + product) per array task, 24 cores each.
#
#   N=$(($(wc -l < SELECT)-1))
#   sbatch --array=0-$((N-1))%30 submit_dftopt_bench.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SCRIPT="$REPO/pipeline/compute/dft_opt_bench.py"
SEL="$REPO/data/raw/screen_v6/dft_sp_r2scan3c/analysis/dftopt_pilot_select.csv"
RESULTS="$REPO/data/raw/screen_v6/dft_sp_r2scan3c/dftopt_bench"
XTB_BIN="/home/schen3/xtb/bin/xtb"; ORCA_BIN="/home/schen3/orca/orca"

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ID=${SLURM_ARRAY_TASK_ID:-0}
TAG=$(printf "row_%03d" "$ID")
OUTDIR="${TMPDIR:-$PROJ/tmp}/dftopt_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$OUTDIR" "$RESULTS/logs"
trap 'rm -rf "$OUTDIR"' EXIT TERM INT

if [[ -f "$RESULTS/${TAG}.csv" ]]; then echo "SKIP $TAG done"; exit 0; fi
echo "DFTOPT $TAG  skip=$ID  node=${SLURMD_NODENAME}  $(date)"

$PY -u "$SCRIPT" \
    --input "$SEL" --output "$OUTDIR/${TAG}.csv" \
    --skip "$ID" --max 1 \
    --method r2SCAN-3c --solvent dmso --orca-solvent DMSO \
    --n-confs 10 --workers 24 \
    --xtb-bin "$XTB_BIN" --orca-bin "$ORCA_BIN" \
    --orca-nprocs 24 --orca-maxcore 2200 \
    --scratch "$OUTDIR" 2>&1 | tee "$RESULTS/logs/${TAG}.log"

# publish only the small CSV (preserve history; never overwrite)
[[ -f "$OUTDIR/${TAG}.csv" ]] && cp "$OUTDIR/${TAG}.csv" "$RESULTS/${TAG}.csv"
echo "FIN $TAG $(date)"
