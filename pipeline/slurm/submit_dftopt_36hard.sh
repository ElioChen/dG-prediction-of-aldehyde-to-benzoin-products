#!/bin/bash
#SBATCH --job-name=dftopt_36hard
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#
# Fair head-to-head: run DIRECT r2SCAN-3c geometry-opt on the SAME 36 hard cases
# that g-xTB-opt converged 36/36 on (gxtb_test/more_mols_hard_20260620.csv:
# 6 each of largeMW/nitro/P/Si/sulfonyl/sulfonylF). The Jun-22 figure compared
# DFT-opt 0/13 (selection.csv) vs g-xTB 36/36 (this set) — DIFFERENT molecule sets.
# This makes it apples-to-apples and persists each ORCA opt input.out for the
# literal SCF-failure string.
#
#   N=36 ; sbatch --array=0-35%12 submit_dftopt_36hard.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SCRIPT="$REPO/pipeline/compute/dft_opt_bench.py"
SEL="$REPO/data/raw/screen_v6/dft_sp_r2scan3c/analysis/dftopt_36hard_select.csv"
# v2 = SERIAL ORCA (nprocs 1, no mpirun). v1 dir kept as evidence of the mpirun-not-found artifact.
RESULTS="$REPO/data/raw/screen_v6/dft_sp_r2scan3c/dftopt_36hard_v3"
XTB_BIN="/home/schen3/xtb/bin/xtb"; ORCA_BIN="/home/schen3/orca/orca"

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OMPI=/gpfs/scratch1/shared/schen3/software/openmpi-4.1.6-install
export PATH="$OMPI/bin:$PATH"
export LD_LIBRARY_PATH="$OMPI/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_rmaps_base_oversubscribe=true OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_btl=self,vader OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_pml=ob1

ID=${SLURM_ARRAY_TASK_ID:-0}
TAG=$(printf "row_%03d" "$ID")
bash "$REPO/pipeline/slurm/clean_node_orphans.sh" 2>/dev/null || true   # self-heal node-local inode quota before generating scratch
OUTDIR="${TMPDIR:-$PROJ/tmp}/dftopt36_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$OUTDIR" "$RESULTS/logs" "$RESULTS/orca_out"
trap 'rm -rf "$OUTDIR"' EXIT TERM INT

if [[ -f "$RESULTS/${TAG}.csv" ]]; then echo "SKIP $TAG done"; exit 0; fi
echo "DFTOPT36 $TAG  row=$ID  node=${SLURMD_NODENAME}  $(date)"

$PY -u "$SCRIPT" \
    --input "$SEL" --output "$OUTDIR/${TAG}.csv" \
    --skip "$ID" --max 1 \
    --method r2SCAN-3c --solvent dmso --orca-solvent DMSO \
    --n-confs 10 --workers 4 \
    --xtb-bin "$XTB_BIN" --orca-bin "$ORCA_BIN" \
    --orca-nprocs 1 --orca-maxcore 3000 \
    --sp-timeout 10800 --opt-timeout 72000 \
    --orca-out-dir "$RESULTS/orca_out" \
    --scratch "$OUTDIR" 2>&1 | tee "$RESULTS/logs/${TAG}.log"

# publish only the small CSV (preserve history; never overwrite)
[[ -f "$OUTDIR/${TAG}.csv" ]] && cp "$OUTDIR/${TAG}.csv" "$RESULTS/${TAG}.csv"
echo "FIN $TAG $(date)"
