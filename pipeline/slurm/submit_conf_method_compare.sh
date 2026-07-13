#!/bin/bash
#SBATCH --job-name=confcmp
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --time=08:00:00
#
# Legacy _rank_conformers vs funnel_v3 conformer search, on the 30 DFT-opt pilot
# molecules: ΔG_xtb / ΔG_orca (r2SCAN-3c//conf) + benzoin broken-topology, per method.
# One molecule per array task, 24 cores.
#   N=$(($(wc -l < SEL)-1)); sbatch --array=0-$((N-1))%30 submit_conf_method_compare.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SCRIPT="$REPO/pipeline/compute/conf_method_compare.py"
SEL="$REPO/data/raw/screen_v6/dft_sp_r2scan3c/analysis/dftopt_pilot_select.csv"
RESULTS="$REPO/data/raw/screen_v6/dft_sp_r2scan3c/conf_method_compare"
XTB_BIN="/home/schen3/xtb/bin/xtb"; ORCA_BIN="/home/schen3/orca/orca"

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ID=${SLURM_ARRAY_TASK_ID:-0}
TAG=$(printf "row_%03d" "$ID")
OUTDIR="${TMPDIR:-$PROJ/tmp}/confcmp_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$OUTDIR" "$RESULTS/logs"
trap 'rm -rf "$OUTDIR"' EXIT TERM INT
if [[ -f "$RESULTS/${TAG}.csv" ]]; then echo "SKIP $TAG done"; exit 0; fi
echo "CONFCMP $TAG  node=${SLURMD_NODENAME}  $(date)"

$PY -u "$SCRIPT" \
    --input "$SEL" --output "$OUTDIR/${TAG}.csv" \
    --skip "$ID" --max 1 \
    --method r2SCAN-3c --solvent dmso --orca-solvent DMSO \
    --n-confs 10 --workers 24 \
    --xtb-bin "$XTB_BIN" --orca-bin "$ORCA_BIN" \
    --orca-nprocs 24 --orca-maxcore 1500 \
    --scratch "$OUTDIR" 2>&1 | tee "$RESULTS/logs/${TAG}.log"

[[ -f "$OUTDIR/${TAG}.csv" ]] && cp "$OUTDIR/${TAG}.csv" "$RESULTS/${TAG}.csv"
echo "FIN $TAG $(date)"
