#!/bin/bash
#SBATCH --job-name=orphan_cron
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/slurm_logs/orphan_cron-%j.out
#
# PERIODIC orphan-scratch cleanup (scrontab/crontab are unavailable on this cluster).
# Self-resubmitting SLURM job: each run cleans dead-job scratch under
# /gpfs/scratch1/nodespecific/schen3.* (clean_orphan_scratch.sh protects LIVE jobs),
# then re-queues itself to run again CADENCE later. Runs on the cheap `staging`
# (I/O) partition so it doesn't burn genoa 24-core units.
#
# Start the chain:   sbatch pipeline/slurm/orphan_cleanup_cron.sh
# Stop it:           scancel --name=orphan_cron     (also cancels the pending next run)
#
REPO="/scratch-shared/schen3/benzoin-dg"
SELF="$REPO/pipeline/slurm/orphan_cleanup_cron.sh"
CADENCE="${ORPHAN_CADENCE:-3hours}"

# Re-queue self FIRST so a cleanup failure can never break the chain.
sbatch --begin="now+${CADENCE}" "$SELF"

echo "[$(date '+%F %T')] orphan cleanup run (cadence=$CADENCE)"
bash "$REPO/pipeline/slurm/clean_orphan_scratch.sh" --delete
echo "[$(date '+%F %T')] done rc=$?"
