#!/bin/bash
#SBATCH --job-name=purge_loop
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/slurm_logs/purge_loop-%j.out
# Loop: delete orphan scratch (idle>30min, jobid NOT live) until inode usage < 70%.
# Fast path (mtime + live-set exclusion, NO slow per-id scontrol). Runs as a job so it
# survives interactive-session teardown. Protects live jobs via the squeue live set + grace.
for iter in $(seq 1 60); do
  pct=$(myquota 2>/dev/null | awk '/wstor_scratch1/{f=1} f&&/Inodes/{gsub("%","",$5);print int($5);exit}')
  echo "[$(date +%T)] iter=$iter inode=${pct}%"
  [ "${pct:-99}" -lt 70 ] && { echo "HEALTHY (<70%) — stop"; break; }
  squeue -u "$USER" -h -o "%i" 2>/dev/null | grep -oE '^[0-9]+' | sort -u > /tmp/live_pl.txt
  find /gpfs/scratch1/nodespecific -maxdepth 2 -type d -name 'schen3.*' -mmin +30 2>/dev/null > /tmp/cand_pl.txt
  awk -v LF=/tmp/live_pl.txt 'FILENAME==LF{live[$1]=1;next}{n=$0;sub(/.*\/schen3\./,"",n);sub(/[._].*/,"",n); if(n!="" && !(n in live)) print $0}' /tmp/live_pl.txt /tmp/cand_pl.txt > /tmp/del_pl.txt
  echo "  deleting $(wc -l < /tmp/del_pl.txt) orphan dirs..."
  xargs -a /tmp/del_pl.txt -d '\n' -P 32 -I{} rm -rf "{}" 2>/dev/null
  sleep 30
done
echo "[$(date +%T)] purge_loop done. final:"; myquota 2>/dev/null | awk '/wstor_scratch1/{f=1} f&&/Inodes/{print $5;exit}'
