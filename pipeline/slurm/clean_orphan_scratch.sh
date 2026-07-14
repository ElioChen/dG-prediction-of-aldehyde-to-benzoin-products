#!/bin/bash
# Safely remove orphaned per-node scratch dirs left by dead SLURM jobs.
#
# Snellius compute jobs create  $TMPDIR/...  ==  /gpfs/scratch1/nodespecific/<node>/schen3.<JobId>
# and are supposed to `rm -rf` it on exit. Cancelled / timed-out jobs skip that
# cleanup, orphaning millions of tiny xtb/Multiwfn files and exhausting the
# inode quota. This script deletes ONLY scratch dirs whose JobId is no longer
# known to SLURM (i.e. the job has left the system) — live jobs are never touched.
#
# Usage:
#   bash ml/clean_orphan_scratch.sh           # dry-run: list what would be deleted
#   bash ml/clean_orphan_scratch.sh --delete  # actually delete
set -euo pipefail

ROOT="/gpfs/scratch1/nodespecific"
DO_DELETE="${1:-}"

# Live per-task JobIds (matches the schen3.<JobId> scratch naming).
# Use BOTH scontrol and squeue; abort if neither SLURM client is usable, so we
# never mistake an unreachable controller for "no live jobs" and nuke real work.
if ! command -v squeue >/dev/null 2>&1 && ! command -v scontrol >/dev/null 2>&1; then
    echo "ABORT: no squeue/scontrol on PATH — cannot determine live jobs."; exit 1
fi
{ scontrol -o show job 2>/dev/null | grep -oP 'JobId=\K[0-9]+' || true
  squeue -u "$USER" -h -o '%i' 2>/dev/null | grep -oE '^[0-9]+' || true
} | sort -u > /tmp/_live_jobids.txt
echo "Live SLURM JobIds: $(wc -l < /tmp/_live_jobids.txt)"

# All scratch dirs owned by this workflow, EXCLUDING anything modified in the last
# GRACE_MIN minutes: a just-started live task may not be in the SLURM dump yet
# (controller throttles under array-start load), and its fresh scratch must be safe.
# '|| true' so find's nonzero exit on other users' unreadable node dirs doesn't trip set -e.
GRACE_MIN="${ORPHAN_GRACE_MIN:-60}"
find "$ROOT" -maxdepth 2 -type d -name 'schen3.*' -mmin +"$GRACE_MIN" 2>/dev/null \
    > /tmp/_all_scratch.txt || true
echo "Total scratch dirs (idle >${GRACE_MIN}min):  $(wc -l < /tmp/_all_scratch.txt)"

# Orphan CANDIDATES = dirs whose NUMERIC JobId is not in the live set.
# Scratch dirs are named  schen3.<JobId>[.<ArrayTask>]  — a running array task's dir is
# schen3.<expandedJobId>.<task>, but SLURM reports only the bare <expandedJobId>. We MUST
# key on the jobid PREFIX (strip the .<task> suffix). The old code keyed on the whole
# "<jobid>.<task>" string, so it flagged EVERY live array task's scratch as an orphan and
# would delete it mid-run (this is the bug behind past whole-run scratch wipes).
# Key the live file on FILENAME (not NR==FNR) so an empty live-jobid file still works.
awk -v LF=/tmp/_live_jobids.txt '
    FILENAME==LF { live[$1]=1; next }
    { n=$0; sub(/.*\/schen3\./,"",n); sub(/[._].*/,"",n);   # n = numeric JobId prefix
      if (n != "" && !(n in live)) print $0 }' \
    /tmp/_live_jobids.txt /tmp/_all_scratch.txt > /tmp/_orphan_candidates.txt
echo "Orphan candidates (jobid not in live set): $(wc -l < /tmp/_orphan_candidates.txt)"

# Final per-id gate: a BULK scontrol dump can be partial under load, so confirm EACH
# candidate's job is really gone before deleting. Only an explicit per-id "Invalid job id"
# is definitive; timeout / throttle / a returned record all mean "maybe alive" -> keep.
#
# PERFORMANCE FIX (2026-07-14): this used to be a SERIAL `while read` loop calling
# scontrol once per candidate. At full-library scale there are 30k-40k+ orphan
# candidates, and a serial ~0.3-1s-per-call scontrol RPC to the controller cannot
# possibly finish within the cron job's 40-minute time limit (~2.5-11 hours needed).
# The result: every single run of orphan_cleanup_cron.sh silently timed out at this
# gate for its entire operational history, deleting NOTHING (orphan-candidate count
# stayed essentially flat across weeks of "successful" 3-hourly resubmissions) -- a
# longstanding no-op, not just a broken chain. Fixed by parallelizing the SAME per-id
# definitive check (no safety weakened, just concurrent instead of serial): export -f +
# xargs -P instead of a background-job pool, so each candidate's result is a single
# atomic `echo` (one write() call, under PIPE_BUF) -- safe to fan into one shared output
# file even at high parallelism.
is_dead() {
    # scontrol exits NON-ZERO for BOTH a timeout AND an invalid job id, so we must NOT
    # gate on the exit code (the old `... || return 1` swallowed the invalid-id case and
    # made this a silent no-op — the reason orphans piled up to the inode quota). Only a
    # timeout (124) means "controller unreachable -> keep"; otherwise inspect the text.
    local out rc
    if out=$(timeout 5 scontrol show job "$1" 2>&1); then rc=0; else rc=$?; fi
    [ "$rc" -eq 124 ] && return 1
    grep -qiE 'Invalid job id' <<<"$out"
}
export -f is_dead
check_one() {
    # jobid = the dir name AFTER the "<user>." prefix, up to the first . or _ (array task
    # suffix). MUST NOT use `grep -oE '[0-9]+' | head -1`: the username "schen3" contains a
    # digit, so that grabbed "3" for EVERY dir — the per-id gate then checked bogus job 3
    # (always "Invalid job id" -> everything "dead"), which would delete live scratch.
    local d="$1" jid
    jid=$(basename "$d"); jid=${jid#*.}; jid=${jid%%[._]*}
    [ -n "$jid" ] || return 0
    is_dead "$jid" && printf '%s\n' "$d"
    return 0
}
export -f check_one
CHECK_PARALLEL="${ORPHAN_CHECK_PARALLEL:-16}"   # moderate: don't hammer the controller
xargs -a /tmp/_orphan_candidates.txt -d '\n' -P "$CHECK_PARALLEL" -I{} bash -c 'check_one "$@"' _ {} \
    > /tmp/_orphan_scratch.txt
N=$(wc -l < /tmp/_orphan_scratch.txt)
echo "Confirmed dead-job orphan dirs: $N"

if [[ "$DO_DELETE" == "--delete" ]]; then
    echo "Deleting $N orphan scratch dirs..."
    xargs -a /tmp/_orphan_scratch.txt -d '\n' -P 8 -I{} rm -rf "{}"
    echo "Done."
else
    echo "(dry-run) Re-run with --delete to remove them. Sample:"
    head -5 /tmp/_orphan_scratch.txt
fi
