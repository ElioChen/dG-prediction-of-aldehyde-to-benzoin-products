#!/bin/bash
#SBATCH --job-name=dft_relbl
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#
# RELABEL the 158 formyl-coupling-bug substrates with the FIXED benzoin generator
# (carbon-CHO-constrained SMARTS in thermo_orca.py, 2026-06-20). Same method as
# dft_sp_r2scan3c_full. Input subset = relabel_smilesfix_158_20260620.csv (158 rows
# of input_v6.csv, flagged no_benzoin_motif by check_smiles.py).
#
#   N=$(($(wc -l < "$SEL")-1)); CH=50; NT=$(( (N+CH-1)/CH ))   # = 4 tasks
#   sbatch --array=0-$((NT-1))%4 submit_dft_sp_relabel158.sh
#   # after it finishes, re-run the SMILES gate to confirm 0 failures:
#   python pipeline/check_smiles.py data/raw/screen_v6/dft_sp_relabel158
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SCRIPT="$REPO/pipeline/compute/thermo_orca.py"
SEL="$REPO/data/raw/screen_v6/relabel_smilesfix_158_20260620.csv"
RESULTS="$REPO/data/raw/screen_v6/dft_sp_relabel158"
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
TAG=$(printf "chunk_%05d" "$ID")
OUTDIR="${TMPDIR:-$PROJ/tmp}/dftrelbl_${SLURM_ARRAY_JOB_ID}_${ID}"   # node-local scratch
mkdir -p "$OUTDIR" "$RESULTS/benzoin_xyz" "$RESULTS/logs"
trap 'rm -rf "$OUTDIR"' EXIT TERM INT

if [[ -f "$RESULTS/${TAG}.csv" ]]; then
    echo "SKIP $TAG (already done) $(date)"; exit 0
fi

echo "DFT-RELABEL $TAG  skip=$SKIP max=$CHUNK  method=$METHOD  node=${SLURMD_NODENAME}  $(date)"
cd "$REPO/pipeline/compute"
$PY -u "$SCRIPT" \
    --input "$SEL" --output-dir "$OUTDIR" \
    --skip "$SKIP" --max "$CHUNK" \
    --conformer default --n-confs 10 --solvent dmso \
    --sp-method "$METHOD" --xtb-bin "$XTB_BIN" --orca-bin "$ORCA_BIN" \
    --orca-nprocs 1 --orca-maxcore 1500 --workers "$WORKERS" \
    --log-level INFO 2>&1 | tee "$RESULTS/logs/${TAG}.log"

if [[ -f "$OUTDIR/delta_G.csv" ]]; then
    cp "$OUTDIR/delta_G.csv" "$RESULTS/${TAG}.csv"
    cp -n "$OUTDIR/benzoin_xyz/"*.xyz "$RESULTS/benzoin_xyz/" 2>/dev/null || true
fi
echo "Done $TAG $(date) -> $RESULTS/${TAG}.csv"
