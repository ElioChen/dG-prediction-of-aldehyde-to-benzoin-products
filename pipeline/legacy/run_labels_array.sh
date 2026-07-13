#!/bin/bash
# Launcher: count molecules in INPUT and submit the per-molecule ORCA array.
#   ./run_labels_array.sh <INPUT.csv> [OUTDIR] [MAX_CONCURRENT]
set -euo pipefail
INPUT="$(readlink -f "${1:?usage: run_labels_array.sh INPUT.csv [OUTDIR] [MAXCONC]}")"
OUTDIR="${2:-/scratch-shared/schen3/benzoin-dg/data/raw/labels}"
MAXC="${3:-32}"
ENSEMBLE_K="${ENSEMBLE_K:-1}"      # export ENSEMBLE_K=5 to Boltzmann-average over conformers
HERE="$(cd "$(dirname "$0")" && pwd)"

N=$(($(wc -l < "$INPUT") - 1))
[[ $N -ge 1 ]] || { echo "no data rows in $INPUT"; exit 1; }
mkdir -p "$OUTDIR/logs"
echo "Submitting $N molecules as array 0-$((N-1))%$MAXC -> $OUTDIR  (ensemble_k=$ENSEMBLE_K)"
sbatch --array=0-$((N-1))%"$MAXC" \
       --export=ALL,INPUT="$INPUT",OUTDIR="$OUTDIR",ENSEMBLE_K="$ENSEMBLE_K" \
       "$HERE/submit_labels_array.sh"
