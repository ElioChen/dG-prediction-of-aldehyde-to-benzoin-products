#!/bin/bash
# Fast per-NODE orphan-scratch self-heal. Deletes THIS node's
# /scratch-local/$USER.<JobId>.* dirs whose SLURM JobId is no longer live.
# Cheap (one node, no cross-node find) — meant to run at the START of every job so
# each node clears dead-job scratch before generating its own. The current job's
# own scratch (JobId is live) is never touched. Complements the periodic global
# orphan_cleanup_cron.sh.
#
# SAFETY (learned the hard way — a bulk `scontrol show job` dump is NOT reliable here):
#   * Under the concurrent job-start load of a big array, the controller throttles and
#     returns an EMPTY or PARTIAL job list. A partial list silently drops live sibling
#     array tasks; acting on it rm -rf's their scratch mid-run (wiped this workflow's
#     product xyz/ dirs across ~1/3 of tasks).
#   * Array tasks are also reported in the compressed `JobId=<master>_<task>` form in
#     some states, so parsing expanded ids out of a bulk dump is fragile.
# Therefore we DO NOT parse a bulk dump. We:
#   1. Skip any dir younger than GRACE_MIN (a live job just created/used it).
#   2. For older candidates, ask SLURM about THAT ONE id and delete ONLY when it answers
#      the definitive "Invalid job id specified". Throttle/timeout/empty -> keep (safe).
set -uo pipefail
LOCAL="/scratch-local"
GRACE_MIN=30                       # cheap pre-filter only; the per-id is_dead() scontrol
                                   # check below is the REAL safety. A 360-min (job-timeout)
                                   # grace was tried and BACKFIRED: it protected recent
                                   # dead-job orphans too, so /scratch-local filled mid-run
                                   # and every task died with Errno 122 (disk quota). 30 min
                                   # is the proven value — a dead job's scratch older than
                                   # that is safe to remove, and live siblings are kept by
                                   # is_dead() regardless of mtime.
[ -d "$LOCAL" ] || exit 0
command -v scontrol >/dev/null 2>&1 || exit 0   # no SLURM client -> do nothing

# True only when SLURM explicitly says the id is unknown (job is really gone). Any other
# outcome (record returned, throttle error, empty) is treated as "maybe alive" -> keep.
is_dead() {
    # scontrol exits NON-ZERO for BOTH a timeout AND an invalid job id. Gating on the exit
    # code (the old `... || return 1`) swallowed the invalid-id case, so is_dead ALWAYS
    # returned false and this self-heal deleted nothing — orphans then filled the inode
    # quota. Distinguish by code: only a timeout (124) is "unreachable -> keep (safe)";
    # for any other outcome, the definitive "Invalid job id" text means the job is gone.
    local jid="$1" out rc
    out=$(timeout 5 scontrol show job "$jid" 2>&1); rc=$?
    [ "$rc" -eq 124 ] && return 1
    grep -qiE 'Invalid job id' <<<"$out"
}

n=0
for d in "$LOCAL/$USER."*; do
    [ -e "$d" ] || continue
    # grace window: skip anything recently modified (a running job's live scratch)
    [ -n "$(find "$d" -maxdepth 0 -mmin -"$GRACE_MIN" 2>/dev/null)" ] && continue
    # jobid = dir name after the "<user>." prefix, up to first . or _ (array-task suffix).
    # NOT `grep -oE '[0-9]+' | head -1`: "schen3" has a digit, so that yielded "3" for every
    # dir and the gate checked bogus job 3 (always invalid -> would delete live scratch).
    jid=$(basename "$d"); jid=${jid#*.}; jid=${jid%%[._]*}
    [ -n "$jid" ] || continue
    is_dead "$jid" || continue          # only delete ids SLURM confirms are gone
    rm -rf "$d" 2>/dev/null && n=$((n+1))
done
[ "$n" -gt 0 ] && echo "[clean_node_orphans] removed $n dead-job scratch dir(s) on $(hostname)"
exit 0
