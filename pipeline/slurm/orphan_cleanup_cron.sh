#!/bin/bash
#SBATCH --job-name=orphan_cron
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/slurm_logs/orphan_cron-%j.out
#
# PERIODIC orphan-scratch cleanup (scrontab/crontab are unavailable on this cluster).
# Self-resubmitting SLURM job: each run cleans dead-job scratch under
# /gpfs/scratch1/nodespecific/schen3.* (clean_orphan_scratch.sh protects LIVE jobs),
# then re-queues itself to run again CADENCE later. Originally ran on the cheap
# `staging` (I/O) partition to avoid burning genoa 24-core units, but switched to
# genoa 2026-07-15 after a single run sat PENDING(Priority) 5+ hours on staging's
# only ~10 nodes (all occupied by other users) -- genoa is far larger and this
# job's own footprint (16 cores) is trivial next to genoa's total capacity.
#
# TIME/CADENCE widened 2026-07-15 (1h/3h -> 4h/6h): clean_orphan_scratch.sh's
# per-id confirm gate is all-or-nothing (writes results to a temp file and only
# enters the delete branch AFTER the full xargs confirm pass finishes) -- a
# walltime kill mid-confirm deletes NOTHING, not a partial cleanup. Under today's
# heavy concurrent cluster load (many other array jobs hammering the same SLURM
# controller), confirming ~23k candidates blew past 1h with zero progress to show
# for it. 4h matches the prior successful oneshot run (job 24637666, 29,899
# candidates, completed and deleted 7,691) at a comparable scale. CADENCE moved
# to 6h (was 3h) so a run taking close to its full 4h walltime can never overlap
# with its own self-resubmitted successor -- two concurrent passes would only add
# MORE simultaneous scontrol load on an already-slow controller, worsening the
# exact problem being fixed.
#
# Start the chain:   sbatch pipeline/slurm/orphan_cleanup_cron.sh
# Stop it:           scancel --name=orphan_cron     (also cancels the pending next run)
#
# 2026-07-14 fixes (chain silently died 2026-07-10, and had been a no-op for its whole
# operational history even before that -- see clean_orphan_scratch.sh's PERFORMANCE FIX
# comment): (1) cpus-per-task 1->16 + time 40min->1h to match the now-parallelized
# per-id confirm gate; (2) the self-resubmission `sbatch` call is now RETRIED and its
# failure is LOGGED LOUDLY instead of silently breaking the chain -- this is exactly
# what happened to job 24527437, whose resubmit apparently failed with no trace in its
# log, and nothing resubmitted it after.
REPO="/scratch-shared/schen3/benzoin-dg"
SELF="$REPO/pipeline/slurm/orphan_cleanup_cron.sh"
CADENCE="${ORPHAN_CADENCE:-6hours}"

# Re-queue self FIRST so a cleanup failure can never break the chain. Retry once on
# failure (transient scheduler hiccups happen); if BOTH attempts fail, say so loudly in
# the log instead of silently letting the chain die with no evidence anywhere.
resubmit_ok=0
for attempt in 1 2; do
    if sbatch --begin="now+${CADENCE}" "$SELF"; then
        resubmit_ok=1; break
    fi
    echo "[$(date '+%F %T')] WARNING: self-resubmit attempt $attempt failed, retrying in 30s"
    sleep 30
done
if [[ "$resubmit_ok" -ne 1 ]]; then
    echo "[$(date '+%F %T')] *** FATAL: orphan_cron self-resubmit FAILED twice -- chain is now BROKEN. ***"
    echo "  Restart manually with: sbatch $SELF"
fi

echo "[$(date '+%F %T')] orphan cleanup run (cadence=$CADENCE)"
export ORPHAN_CHECK_PARALLEL="${ORPHAN_CHECK_PARALLEL:-16}"
bash "$REPO/pipeline/slurm/clean_orphan_scratch.sh" --delete
echo "[$(date '+%F %T')] done rc=$?"
