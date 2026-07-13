#!/bin/bash
#SBATCH --job-name=dftsp_retry
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_funnelv3/retry7200/logs/dftsp_retry_%A_%a.out
#
# RETRY pass for the ~4% of full-library DFT-SP molecules that hit the 3600s ORCA
# soft-timeout in job 24178884. Identical settings EXCEPT:
#   * TIMEOUT=7200 (2h) -- the script's own documented safe floor; 3600 killed legit-large SPs
#   * MAN  = manifest_retry.parquet  (only the failed ids; build with build_retry_manifest.py)
#   * RESULTS = .../retry7200/        (separate dir -> never overwrites the original chunk CSVs)
# Geometry-consistent: same saved funnel_v3 xyz + xTB RRHO thermal as the main run.
#
#   python pipeline/compute/build_retry_manifest.py        # after main array drains
#   N=$(python -c "import pandas as pd;print(len(pd.read_parquet(
#       '/scratch-shared/schen3/benzoin-dg/data/raw/dft_sp_funnelv3/manifest_retry.parquet')))")
#   CH=96; NT=$(( (N+CH-1)/CH ))
#   sbatch --array=0-$((NT-1))%128 pipeline/slurm/submit_dft_sp_retry.sh
set -o pipefail
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
MAN="$REPO/data/raw/dft_sp_funnelv3/manifest_retry.parquet"
RESULTS="$REPO/data/raw/dft_sp_funnelv3/retry7200"
CHUNK="${CHUNK:-96}"; WORKERS="${WORKERS:-48}"; MAXCORE="${MAXCORE:-1500}"; TIMEOUT="${TIMEOUT:-7200}"
ORCA_BIN="/home/schen3/orca/orca"
mkdir -p "$RESULTS/logs"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/orca:$PATH"
export LD_LIBRARY_PATH="/home/schen3/orca:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ID=${SLURM_ARRAY_TASK_ID:-0}; SKIP=$(( ID * CHUNK )); TAG=$(printf "chunk_%05d" "$ID")
[[ -f "$RESULTS/${TAG}.csv" ]] && { echo "SKIP $TAG (done)"; exit 0; }

echo "DFTSP-retry $TAG skip=$SKIP chunk=$CHUNK workers=$WORKERS timeout=$TIMEOUT node=${SLURMD_NODENAME} $(date)"
$PY -u "$REPO/pipeline/compute/dft_sp_from_geom.py" \
    --manifest "$MAN" --skip "$SKIP" --max "$CHUNK" \
    --out-csv "$RESULTS/${TAG}.csv" --workers "$WORKERS" --maxcore "$MAXCORE" \
    --timeout "$TIMEOUT" --orca-bin "$ORCA_BIN"
echo "Done $TAG $(date)"
