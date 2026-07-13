#!/bin/bash
# Launcher: chunk INPUT and submit the descriptor array.
#   ./run_descriptors_array.sh <INPUT.csv> [OUTDIR] [CHUNK_SIZE]
set -euo pipefail
INPUT="$(readlink -f "${1:?usage: run_descriptors_array.sh INPUT.csv [OUTDIR] [CHUNK]}")"
OUTDIR="${2:-/scratch-shared/schen3/benzoin-dg/data/raw/descriptors}"
CHUNK="${3:-30}"
HERE="$(cd "$(dirname "$0")" && pwd)"

N=$(($(wc -l < "$INPUT") - 1))
[[ $N -ge 1 ]] || { echo "no data rows in $INPUT"; exit 1; }
NCH=$(( (N + CHUNK - 1) / CHUNK ))
mkdir -p "$OUTDIR/logs"
echo "Submitting $N molecules in $NCH chunks of $CHUNK -> $OUTDIR"
sbatch --array=0-$((NCH-1)) \
       --export=ALL,INPUT="$INPUT",OUTDIR="$OUTDIR",CHUNK="$CHUNK" \
       "$HERE/submit_descriptors_array.sh"
