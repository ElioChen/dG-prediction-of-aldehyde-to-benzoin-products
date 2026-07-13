#!/bin/bash
#SBATCH --job-name=gxtb_dft1pct
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch-shared/schen3/gxtb_test/logs/gxtbdft1pct_%A_%a.out
#
# FULL 1% (2192 mols): g-xTB geom -> g-xTB Hessian thermo + r2SCAN-3c SP, for
# ΔG_gxtb and ΔG_dft//gxtb vs the existing ΔG_dft//xtb. Same driver/method as the
# pilot. CHUNK molecules per array task, serial ORCA (nproc 1) x --workers across mols.
#   N=$(($(wc -l < "$SEL")-1)); CH=50; NT=$(( (N+CH-1)/CH ))   # = 44 tasks
#   sbatch --array=0-$((NT-1))%8 submit_gxtbgeom_dftsp_1pct.sh
#
PROJ="/scratch-shared/schen3"; WD="$PROJ/gxtb_test"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SEL="$WD/full1pct_gxtbgeom_20260620.csv"
RES="$WD/gxtbgeom_dftsp_1pct"
CHUNK="${CHUNK:-50}"
source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ID=${SLURM_ARRAY_TASK_ID:-0}; SKIP=$(( ID * CHUNK ))
TAG=$(printf "chunk_%05d" "$ID")
mkdir -p "$RES/xyz" "$RES/logs"
[[ -f "$RES/${TAG}.csv" ]] && { echo "SKIP $TAG done"; exit 0; }

echo "gxtb->DFT 1% $TAG skip=$SKIP max=$CHUNK node=${SLURMD_NODENAME} $(date)"
cd "$WD"
$PY -u gxtbgeom_dftsp.py \
    --input "$SEL" --out "$RES/${TAG}.csv" --xyzdir "$RES/xyz" \
    --scratch "${TMPDIR:-$PROJ/tmp}" --skip "$SKIP" --max "$CHUNK" \
    --nproc 1 --workers 24
echo "Done $TAG $(date)"
