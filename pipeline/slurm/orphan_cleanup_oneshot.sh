#!/bin/bash
#SBATCH --job-name=orphan_oneshot
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/slurm_logs/orphan_oneshot-%j.out
# One-shot, long-walltime run of the blessed orphan cleaner so the per-id confirm loop
# completes (the interactive nohup kept dying). Protects live jobs (live-set + grace + per-id).
REPO="/scratch-shared/schen3/benzoin-dg"
echo "[$(date '+%F %T')] start one-shot orphan cleanup"
bash "$REPO/pipeline/slurm/clean_orphan_scratch.sh" --delete
echo "[$(date '+%F %T')] done rc=$?"
