#!/bin/bash
#SBATCH --job-name=orphan_oneshot
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/slurm_logs/orphan_oneshot-%j.out
# One-shot, long-walltime catch-up run of the orphan cleaner. cpus-per-task bumped to 16
# (2026-07-14) to match clean_orphan_scratch.sh's new parallel per-id confirm gate
# (ORPHAN_CHECK_PARALLEL, default 16) -- the old serial version of that gate could not
# finish even in 4h once orphan candidates passed a few thousand; the recurring
# orphan_cleanup_cron.sh chain (40min budget) had been timing out at this same gate for
# its entire operational history without deleting anything, and broke outright on
# 2026-07-10 when a resubmission attempt silently failed. Run this once to catch up on
# the backlog (~41k candidates as of 2026-07-14), then re-arm the chain.
REPO="/scratch-shared/schen3/benzoin-dg"
echo "[$(date '+%F %T')] start one-shot orphan cleanup"
export ORPHAN_CHECK_PARALLEL="${ORPHAN_CHECK_PARALLEL:-16}"
bash "$REPO/pipeline/slurm/clean_orphan_scratch.sh" --delete
echo "[$(date '+%F %T')] done rc=$?"
