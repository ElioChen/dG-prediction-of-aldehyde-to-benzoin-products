#!/bin/bash
# Launcher: count molecules and submit the unified featurize array.
#   ENSEMBLE_K=5 ./run_featurize_array.sh <INPUT.csv> [OUTDIR] [MAX_CONCURRENT]
set -euo pipefail
INPUT="$(readlink -f "${1:?usage: run_featurize_array.sh INPUT.csv [OUTDIR] [MAXCONC]}")"
OUTDIR="${2:-/scratch-shared/schen3/benzoin-dg/data/raw/featurize}"
MAXC="${3:-50}"
ENSEMBLE_K="${ENSEMBLE_K:-5}"
HERE="$(cd "$(dirname "$0")" && pwd)"

N=$(($(wc -l < "$INPUT") - 1))
[[ $N -ge 1 ]] || { echo "no data rows in $INPUT"; exit 1; }
mkdir -p "$OUTDIR/logs"
echo "Submitting $N molecules as array 0-$((N-1))%$MAXC -> $OUTDIR  (ensemble_k=$ENSEMBLE_K)"
sbatch --array=0-$((N-1))%"$MAXC" \
       --export=ALL,INPUT="$INPUT",OUTDIR="$OUTDIR",ENSEMBLE_K="$ENSEMBLE_K" \
       "$HERE/submit_featurize_array.sh"
