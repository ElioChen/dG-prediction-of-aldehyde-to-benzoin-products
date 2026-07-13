#!/bin/bash
#SBATCH --job-name=dft_sp
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#
# DFT single-point ΔG on the 1% selection, SAME base method as the 220k screen
# (_rank_conformers ETKDG+GFN2-xTB, n_confs 10, DMSO, xTB-ohess) + r2SCAN-3c//xTB
# single point. Saves reactant & product G (xTB and DFT) and the product xyz.
# One array task = a slice of CHUNK molecules (via --skip/--max).
#
#   SEL=/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/analysis/dft_selection_v2_input.csv
#   N=$(($(wc -l < "$SEL")-1)); CH=50; NT=$(( (N+CH-1)/CH ))
#   sbatch --array=0-$((NT-1))%32 submit_dft_sp.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SCRIPT="$REPO/pipeline/compute/thermo_orca.py"
SEL="$REPO/data/raw/screen_v6/analysis/dft_selection_v2_input.csv"
RESULTS="$REPO/data/raw/screen_v6/dft_sp_r2scan3c"
CHUNK="${CHUNK:-50}"
METHOD="${METHOD:-r2SCAN-3c}"

XTB_BIN="/home/schen3/xtb/bin/xtb"; ORCA_BIN="/home/schen3/orca/orca"
WORKERS="${WORKERS:-24}"

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ID=${SLURM_ARRAY_TASK_ID:-0}
SKIP=$(( ID * CHUNK ))
TAG=$(printf "chunk_%04d" "$ID")
OUTDIR="${TMPDIR:-$PROJ/tmp}/dft_${SLURM_ARRAY_JOB_ID}_${ID}"   # node-local scratch
mkdir -p "$OUTDIR" "$RESULTS/benzoin_xyz" "$RESULTS/logs"
trap 'rm -rf "$OUTDIR"' EXIT TERM INT

echo "DFT-SP $TAG  skip=$SKIP max=$CHUNK  method=$METHOD  node=${SLURMD_NODENAME}  $(date)"
cd "$REPO/pipeline/compute"
$PY -u "$SCRIPT" \
    --input "$SEL" --output-dir "$OUTDIR" \
    --skip "$SKIP" --max "$CHUNK" \
    --conformer default --n-confs 10 --solvent dmso \
    --sp-method "$METHOD" --xtb-bin "$XTB_BIN" --orca-bin "$ORCA_BIN" \
    --orca-nprocs 1 --orca-maxcore 1500 --workers "$WORKERS" \
    --log-level INFO 2>&1 | tee "$RESULTS/logs/${TAG}.log"

# keep only the key outputs (CSV + product geometries); drop bulk ORCA/xtb scratch
if [[ -f "$OUTDIR/delta_G.csv" ]]; then
    cp "$OUTDIR/delta_G.csv" "$RESULTS/${TAG}.csv"
    cp -n "$OUTDIR/benzoin_xyz/"*.xyz "$RESULTS/benzoin_xyz/" 2>/dev/null || true
fi
echo "Done $TAG $(date) -> $RESULTS/${TAG}.csv"
