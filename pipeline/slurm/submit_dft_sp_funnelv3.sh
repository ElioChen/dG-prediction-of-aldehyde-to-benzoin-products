#!/bin/bash
#SBATCH --job-name=dftsp_fv3
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G        # >344/3 GB so only ~2 tasks pack per genoa node (96 SPs/node, not 192)
                          # -> less memory-bandwidth contention; bills ~1.5x cores-equiv (ample budget)
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_funnelv3/logs/dftsp_%A_%a.out
#
# DRAFT — genoa-tuned full-library r2SCAN-3c ΔG by SP on the SAVED funnel_v3 geometries
# (pipeline/compute/dft_sp_from_geom.py). Geometry-consistent with the g-xTB/GFN2 library;
# skips conformer search + xTB-opt; reuses xTB RRHO thermal.
#
# genoa tuning rationale:
#   * cpus-per-task=48  -> 2 billing units of 24 (genoa bills in 24-core units; avoid odd sizes)
#   * WORKERS=48, orca-nprocs=1  -> 48 molecules in parallel, 1-core ORCA each (SPs scale poorly
#     past a few cores at this molecule size; 1-core x N parallel = best core-h efficiency)
#   * default mem for 48c ~86G -> ~1.8 GB/worker (> the proven 1 GB/worker run); maxcore 1500
#   * %128 is the genoa QOS MaxJobsPU CEILING (can't raise to %192); speed comes from cores/task,
#     so 48c/task = 128*48 = 6144 cores in flight (~2x the old 24c run) -> ~18-19 h wall.
#     For ~9 h, bump cpus-per-task/WORKERS to 96 (3 units) IF enough genoa nodes are free.
#   * node-local scratch + per-molecule rmtree; keep ONLY the result CSV -> inode-safe.
#
#   N=$(python -c "import pandas as pd;print(len(pd.read_parquet('$MAN')))")
#   CH=96; NT=$(( (N+CH-1)/CH ))
#   sbatch --array=0-$((NT-1))%128 pipeline/slurm/submit_dft_sp_funnelv3.sh
#
# PRE-LAUNCH: build manifest once + SMOKE TEST (reproduce a few dft_sp_r2scan3c_full values):
#   python pipeline/compute/dft_sp_from_geom.py --out-csv /tmp/smoke.csv --smoke --workers 3
set -o pipefail   # NOT -u: `source /etc/profile` / `module load` reference unset vars and crash under -u
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"        # prod env with ORCA bindings
MAN="$REPO/data/raw/dft_sp_funnelv3/manifest.parquet"
RESULTS="$REPO/data/raw/dft_sp_funnelv3"
CHUNK="${CHUNK:-96}"; WORKERS="${WORKERS:-48}"; MAXCORE="${MAXCORE:-1500}"; TIMEOUT="${TIMEOUT:-7200}"
# TIMEOUT 7200s = pure insurance: omit-SCF (default) had 0 timeouts at 2h, and some legit-large
# SPs take >1h, so don't shorten below ~2h or you kill valid molecules. (TightSCF was the hang.)
ORCA_BIN="/home/schen3/orca/orca"
mkdir -p "$RESULTS/logs"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ID=${SLURM_ARRAY_TASK_ID:-0}; SKIP=$(( ID * CHUNK )); TAG=$(printf "chunk_%05d" "$ID")
[[ -f "$RESULTS/${TAG}.csv" ]] && { echo "SKIP $TAG (done)"; exit 0; }   # resume-safe

echo "DFTSP-fv3 $TAG skip=$SKIP chunk=$CHUNK workers=$WORKERS node=${SLURMD_NODENAME} $(date)"
$PY -u "$REPO/pipeline/compute/dft_sp_from_geom.py" \
    --manifest "$MAN" --skip "$SKIP" --max "$CHUNK" \
    --out-csv "$RESULTS/${TAG}.csv" --workers "$WORKERS" --maxcore "$MAXCORE" \
    --timeout "$TIMEOUT" --orca-bin "$ORCA_BIN"
echo "Done $TAG $(date)"
