#!/bin/bash
#SBATCH --job-name=gxtb_solv_camp
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --time=24:00:00
#
# Same-molecule, fully-solvated benzoin ΔG on the 1% pilot (2192), BOTH geometry routes
# (funnel-geom Option A + g-xTB-cosmo-opt Option B). SERIAL ORCA (nprocs 1, NO mpirun —
# avoids the OpenMPI-4.1.6-vs-4.1.5 / mpirun-not-found artifact that killed the parallel runs).
# CHUNK molecules per array task.
#
#   N=2192 ; CHUNK=24 ; NT=ceil(N/CHUNK)=92 ; molecule-level parallel (WORKERS=24), NO MPI
#   sbatch --array=0-91%128 submit_gxtb_solv_campaign.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SCRIPT="$REPO/pipeline/compute/gxtb_solv_campaign.py"
SEL="$REPO/data/raw/screen_v6/dft_sp_r2scan3c/analysis/pilot_2191_select.csv"
RESULTS="$REPO/data/raw/screen_v6/dft_sp_r2scan3c/gxtb_solv_pilot"
CHUNK=24

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
export XTB_BIN="/home/schen3/xtb/bin/xtb"
export GXTB_BIN="/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb"
export ORCA_BIN="/home/schen3/orca/orca"
export XTBPATH="/home/schen3/xtb/share/xtb"
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ORCA_NPROCS=1 ORCA_MAXCORE=1500 WORKERS=12   # molecule-level parallel, serial ORCA each (no MPI)

ID=${SLURM_ARRAY_TASK_ID:-0}
TAG=$(printf "chunk_%04d" "$ID")
SKIP=$(( ID * CHUNK ))
bash "$REPO/pipeline/slurm/clean_node_orphans.sh" 2>/dev/null || true   # self-heal node-local inode quota before generating scratch
OUTDIR="${TMPDIR:-$PROJ/tmp}/solvcamp_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$OUTDIR" "$RESULTS/logs"
trap 'rm -rf "$OUTDIR"' EXIT TERM INT

if [[ -f "$RESULTS/${TAG}.csv" ]]; then echo "SKIP $TAG done"; exit 0; fi
echo "SOLVCAMP $TAG  skip=$SKIP max=$CHUNK  node=${SLURMD_NODENAME}  $(date)"

$PY -u "$SCRIPT" \
    --input "$SEL" --output "$OUTDIR/${TAG}.csv" \
    --skip "$SKIP" --max "$CHUNK" --workers 12 \
    --scratch "$OUTDIR" 2>&1 | tee "$RESULTS/logs/${TAG}.log"

[[ -f "$OUTDIR/${TAG}.csv" ]] && cp "$OUTDIR/${TAG}.csv" "$RESULTS/${TAG}.csv"
echo "FIN $TAG $(date)"
