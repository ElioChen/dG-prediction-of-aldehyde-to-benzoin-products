#!/bin/bash
#SBATCH --job-name=clean_orphans
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/clean_orphans_%j.out
#
# Account-wide cleanup of orphaned per-node scratch work directories.
#
# /scratch-local -> /gpfs/scratch1/nodespecific/<node>, on the scratch1 GPFS filesystem,
# counts against schen3's scratch1 INODE quota. Tasks hard-killed / preempted /
# node-failed leave $TMPDIR (`<node>/schen3.<jobid>[_<taskid>]`, Snellius
# `$USER.$SLURM_JOB_ID` convention) behind, full of xtb / Multiwfn per-call scratch.
# 2026-09-02..03: quota hit ~116% of the 4,000,000 hard cap across BDE + cross-benzoin +
# NHC jobs -> every job about to fail on write. User approved an account-wide sweep and
# a persistent looping janitor.
#
# KEEPS  a workdir iff its <jobid> OR <jobid>_<taskid> OR raw per-task JobID is currently
#        in `squeue`, OR it was modified in the last KEEP_MIN minutes (launch/race guard).
# DELETES everything else, in parallel (xargs -P NPROC).
#
#   DRY=1 LOOP=0 bash pipeline/bde/clean_orphan_nodespec_workdirs.sh   # report only
#   sbatch pipeline/bde/clean_orphan_nodespec_workdirs.sh              # persistent janitor
set -o pipefail

USER_NAME="${USER:-schen3}"
NODESPEC="${NODESPEC:-/gpfs/scratch1/nodespecific}"
KEEP_MIN="${KEEP_MIN:-25}"
DRY="${DRY:-0}"
LOOP="${LOOP:-1}"
SLEEP_SEC="${SLEEP_SEC:-900}"
NPROC="${NPROC:-16}"

alive_file="$(mktemp)"
trap 'rm -f "$alive_file"' EXIT

build_alive() {
  : > "$alive_file"
  squeue -u "$USER_NAME" -h -r -o "%i" 2>/dev/null | sed 's/_.*//'  >> "$alive_file"
  squeue -u "$USER_NAME" -h -r -o "%i" 2>/dev/null                  >> "$alive_file"
  squeue -u "$USER_NAME" -h -r -O JobID 2>/dev/null | tr -dc '0-9_\n' | grep . >> "$alive_file"
  sort -u -o "$alive_file" "$alive_file"
}

sweep() {
  build_alive
  local nalive; nalive=$(wc -l < "$alive_file")
  local list; list=$(mktemp)
  find "$NODESPEC" -mindepth 2 -maxdepth 2 -user "$USER_NAME" -type d -name 'schen3.*' 2>/dev/null | \
  while IFS= read -r d; do
    b=$(basename "$d"); id=${b#schen3.}; jid=${id%%_*}
    grep -qxF "$id" "$alive_file" && continue
    grep -qxF "$jid" "$alive_file" && continue
    [ -n "$(find "$d" -maxdepth 0 -mmin -"$KEEP_MIN" 2>/dev/null)" ] && continue
    printf '%s\0' "$d"
  done > "$list"
  local n; n=$(tr -cd '\0' < "$list" | wc -c)
  if [ "$DRY" = 1 ]; then
    echo "$(date '+%m-%d %H:%M') DRY: alive_ids=$nalive would_remove=$n"
    tr '\0' '\n' < "$list" | head -20
  else
    xargs -0 -r -P "$NPROC" -n 1 rm -rf < "$list"
    echo "$(date '+%m-%d %H:%M') sweep: alive_ids=$nalive removed=$n"
  fi
  rm -f "$list"
}

echo "clean_orphans start $(date) DRY=$DRY LOOP=$LOOP KEEP_MIN=$KEEP_MIN NPROC=$NPROC"
while true; do
  sweep
  [ "$LOOP" = 1 ] && [ "$DRY" != 1 ] || break
  sleep "$SLEEP_SEC"
done
