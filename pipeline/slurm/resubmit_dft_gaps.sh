#!/bin/bash
# resubmit_dft_gaps.sh
# Fill-gap resubmit for the dft_full array (submit_dft_sp_full.sh).
# Scans the results dir for any chunk_XXXXX.csv missing in 0..NT-1 and
# resubmits ONLY those task IDs, with bumped mem+time so OOM/TIMEOUT
# stragglers (big/slow molecules) succeed. The base submit script has a
# resume guard (skips chunks whose CSV already exists), so this is safe to
# run repeatedly and cannot clobber finished work.
#
# Usage:
#   bash resubmit_dft_gaps.sh            # dry-run: just list missing chunks
#   bash resubmit_dft_gaps.sh --submit   # actually sbatch the gap list
#
# Tunables (env override): MEM, TIME, THROTTLE
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
SEL="$REPO/data/raw/screen_v6/input_v6.csv"
RES="$REPO/data/raw/screen_v6/dft_sp_r2scan3c_full"
SUBMIT="/gpfs/scratch1/shared/schen3/benzoin-dg/pipeline/slurm/submit_dft_sp_full.sh"
CHUNK=50
MEM="${MEM:-64G}"          # base run used 24G; max observed MaxRSS 16.8G, rare spikes OOM'd
TIME="${TIME:-12:00:00}"   # base run used 8h; slowest chunks hit ~7.5h -> give headroom
THROTTLE="${THROTTLE:-64}"

N=$(($(wc -l < "$SEL")-1)); NT=$(( (N+CHUNK-1)/CHUNK ))

missing=()
for ((i=0; i<NT; i++)); do
    printf -v tag "chunk_%05d" "$i"
    [[ -f "$RES/${tag}.csv" ]] || missing+=("$i")
done

if [[ ${#missing[@]} -eq 0 ]]; then
    echo "No missing chunks (0..$((NT-1))). Full array complete."
    exit 0
fi

# Compress to a comma list (sbatch accepts explicit IDs).
list=$(IFS=,; echo "${missing[*]}")
echo "NT=$NT  missing=${#missing[@]} chunks"
echo "array list: $list"
echo "would run: sbatch --array=${list}%${THROTTLE} --mem=$MEM --time=$TIME $SUBMIT"

if [[ "${1:-}" == "--submit" ]]; then
    jid=$(sbatch --parsable --array="${list}%${THROTTLE}" --mem="$MEM" --time="$TIME" "$SUBMIT")
    echo "Submitted gap-fill array: $jid"
fi
