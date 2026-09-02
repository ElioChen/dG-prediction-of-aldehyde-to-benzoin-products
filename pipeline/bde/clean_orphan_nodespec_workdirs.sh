#!/bin/bash
#SBATCH --job-name=clean_orphans
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/clean_orphans_%j.out
#
# Account-wide cleanup of orphaned per-node scratch work directories.
#
# /scratch-local -> /gpfs/scratch1/nodespecific/<node>, which lives on the scratch1
# GPFS filesystem and counts against schen3's scratch1 INODE quota. Tasks that are
# hard-killed / preempted / node-failed leave their $TMPDIR
# (`<node>/schen3.<jobid>[_<taskid>]`, Snellius `$USER.$SLURM_JOB_ID` convention)
# behind, full of xtb / Multiwfn per-call scratch files. 2026-09-02: the quota hit
# 105% of the 4,000,000 hard cap across BDE + cross-benzoin + NHC jobs -> every job
# about to fail on write. User approved an account-wide sweep.
#
# KEEPS  a workdir iff its <jobid> OR <jobid>_<taskid> is currently in `squeue`,
#        OR it was modified in the last KEEP_MIN minutes (launch/race guard).
# DELETES everything else under /gpfs/scratch1/nodespecific/*/schen3.*
#
#   DRY=1 bash pipeline/bde/clean_orphan_nodespec_workdirs.sh   # report only
#   DRY=0 sbatch pipeline/bde/clean_orphan_nodespec_workdirs.sh # loop-clean on staging
set -o pipefail

USER_NAME="${USER:-schen3}"
NODESPEC="${NODESPEC:-/gpfs/scratch1/nodespecific}"
KEEP_MIN="${KEEP_MIN:-20}"
DRY="${DRY:-0}"
LOOP="${LOOP:-1}"          # 1 = keep sweeping every SLEEP_SEC; 0 = one pass
SLEEP_SEC="${SLEEP_SEC:-1200}"
COUNT_INODES="${COUNT_INODES:-0}"   # 1 = tally freed inodes (slower)

sweep() {
  local alive_n orphans=0 kept=0 freed=0
  declare -A ALIVE
  # (a) array form  <arrayjob>_<task>  and its bare <arrayjob>
  while IFS= read -r j; do
    j=${j%%+*}
    [[ -n "$j" ]] || continue
    ALIVE["$j"]=1
    ALIVE["${j%%_*}"]=1
  done < <(squeue -u "$USER_NAME" -h -r -o "%i" 2>/dev/null)
  # (b) RAW per-task JobIDs (Snellius gives each array task its own numeric JobID, and
  #     some jobs name their /scratch-local dir schen3.<rawjobid> not schen3.<arrayjob>_<task>)
  while IFS= read -r j; do
    j=$(echo "$j" | tr -dc '0-9_')
    [[ -n "$j" ]] && ALIVE["$j"]=1
  done < <(squeue -u "$USER_NAME" -h -r -O JobID 2>/dev/null)
  alive_n=${#ALIVE[@]}

  while IFS= read -r d; do
    [[ -n "$d" ]] || continue
    local base id jid
    base=$(basename "$d")
    id=${base#schen3.}
    jid=${id%%_*}
    if [[ -n "${ALIVE[$id]}" || -n "${ALIVE[$jid]}" ]]; then kept=$((kept+1)); continue; fi
    if [[ -n "$(find "$d" -maxdepth 0 -mmin -"$KEEP_MIN" 2>/dev/null)" ]]; then kept=$((kept+1)); continue; fi
    if [[ "$COUNT_INODES" == 1 ]]; then
      local c; c=$(find "$d" -xdev 2>/dev/null | wc -l); freed=$((freed+c))
    fi
    if [[ "$DRY" == 1 ]]; then echo "WOULD rm $d"; else rm -rf "$d"; fi
    orphans=$((orphans+1))
  done < <(find "$NODESPEC" -mindepth 2 -maxdepth 2 -user "$USER_NAME" -type d -name 'schen3.*' 2>/dev/null)

  echo "$(date '+%m-%d %H:%M') sweep: alive_ids=$alive_n orphans_$([[ $DRY == 1 ]] && echo would_ )removed=$orphans kept=$kept${COUNT_INODES:+ inodes_freed~=$freed}"
}

echo "clean_orphans start $(date) DRY=$DRY LOOP=$LOOP KEEP_MIN=$KEEP_MIN"
while true; do
  sweep
  [[ "$LOOP" == 1 && "$DRY" != 1 ]] || break
  sleep "$SLEEP_SEC"
done
