#!/bin/bash
#SBATCH --job-name=purge_par
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/slurm_logs/purge_par-%A_%a.out
# Multi-task parallel orphan delete: partition the tcn node-dirs by array index so N tasks
# delete different nodes' scratch concurrently (more unlink firepower to the GPFS MDS).
# Protects live jobs via squeue live-set + 30min mtime grace.
NT=${SLURM_ARRAY_TASK_COUNT:-20}; ID=${SLURM_ARRAY_TASK_ID:-0}
{ scontrol -o show job 2>/dev/null | grep -oP 'JobId=\K[0-9]+'; squeue -u "$USER" -h -o "%i" 2>/dev/null | grep -oE '^[0-9]+'; } | sort -u > /tmp/live_pp_$ID.txt   # EXPANDED array-task ids included
i=0
for nodedir in /gpfs/scratch1/nodespecific/*/; do
  i=$((i+1)); [ $(( i % NT )) -ne "$ID" ] && continue   # this task owns every NT-th node dir
  find "$nodedir" -maxdepth 1 -type d -name 'schen3.*' -mmin +15 2>/dev/null \
    | awk -v LF=/tmp/live_pp_$ID.txt 'FILENAME==LF{live[$1]=1;next}{n=$0;sub(/.*\/schen3\./,"",n);sub(/[._].*/,"",n); if(n!="" && !(n in live)) print $0}' /tmp/live_pp_$ID.txt - \
    | xargs -d '\n' -P 16 -I{} rm -rf "{}" 2>/dev/null
done
echo "[$(date +%T)] task $ID done"
