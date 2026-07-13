#!/bin/bash
#SBATCH --job-name=post_dft_chain
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/post_dft_chain_%j.out
#
# Auto-fires when the full DFT-SP array 24178884 DRAINS (submit with
#   sbatch --dependency=afterany:24178884 pipeline/slurm/post_dft_chain.sh ).
# Then: (1) build retry manifest for the 3600s ORCA timeouts + submit 7200s retry array;
#       (2) finalize the champion g-xTB->DFT correction on the now-complete labels.
set -o pipefail
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"     # pandas+sklearn+xgboost+joblib
MANR="$REPO/data/raw/dft_sp_funnelv3/manifest_retry.parquet"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "=== POST-DFT CHAIN start $(date) ==="

echo "--- [1/3] build retry manifest (3600s timeouts + gaps) ---"
$PY "$REPO/pipeline/compute/build_retry_manifest.py" || echo "build_retry_manifest failed"

echo "--- [2/3] submit 7200s retry array (if any missing) ---"
if [[ -f "$MANR" ]]; then
  N=$($PY -c "import pandas as pd;print(len(pd.read_parquet('$MANR')))" 2>/dev/null || echo 0)
  echo "retry molecules: $N"
  if [[ "$N" -gt 0 ]]; then
    CH=96; NT=$(( (N + CH - 1) / CH ))
    cd "$REPO" && sbatch --array=0-$((NT-1))%128 pipeline/slurm/submit_dft_sp_retry.sh && echo "submitted retry array NT=$NT"
  else
    echo "no missing molecules — skip retry"
  fi
fi

echo "--- [3/3] finalize champion correction on complete labels ---"
$PY "$REPO/pipeline/analysis/finalize_correction.py" || echo "finalize failed"

echo "=== POST-DFT CHAIN done $(date) ==="
