#!/bin/bash
# Long-running watch for the full-library cb array. Polls every POLL sec; exits (and so
# re-notifies the agent) on: array drained (DONE), any quota error (ANOMALY), or a large
# cluster of FAILED/TIMEOUT/OOM tasks (ANOMALY). Writes progress to STATUS every poll so the
# outcome is durable even if this process dies (the SLURM job runs independently regardless).
set -uo pipefail
REPO="/scratch-shared/schen3/benzoin-dg"
JOB=24128375
OUT="$REPO/data/cross_benzoin/homo_v6"
NCHUNK=2208
POLL=1800                 # 30 min
FAIL_ALERT=30             # alert if >= this many tasks FAILED/TIMEOUT/OOM (isolated fails are normal)
STATUS="$REPO/cross_benzoin/slurm/FULLLIB_WATCH_STATUS.log"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

say "=== fulllib watch armed: job $JOB, $NCHUNK chunks, poll ${POLL}s ==="
for i in $(seq 1 200); do          # up to ~100h
  running=$(squeue -j "$JOB" -h 2>/dev/null | wc -l)
  done_chunks=$(ls "$OUT"/chunk_*/products.csv 2>/dev/null | wc -l)
  # scope to THIS job's logs only — the dir also holds an old run's (23939656) logs.
  qerr=$(grep -l 'Errno 122\|Disk quota' "$OUT"/logs/cb_${JOB}_*.out 2>/dev/null | wc -l)
  failed=$(sacct -j "$JOB" -X -n --format=State 2>/dev/null | grep -ciE 'FAILED|TIMEOUT|OUT_OF_ME|NODE_FAIL')
  say "poll $i: running=$running done_chunks=$done_chunks/$NCHUNK fail_tasks=$failed quota_logs=$qerr"

  if [ "$qerr" -gt 0 ]; then
    say "=== ANOMALY: quota errors in $qerr log(s) — investigate (per-mol rmtree should prevent this) ==="; exit 2
  fi
  if [ "$failed" -ge "$FAIL_ALERT" ]; then
    say "=== ANOMALY: $failed tasks FAILED/TIMEOUT/OOM (>= $FAIL_ALERT) — investigate ==="; exit 3
  fi
  if [ "$running" -eq 0 ] && [ "$i" -gt 1 ]; then
    say "=== DONE: array drained. done_chunks=$done_chunks/$NCHUNK fail_tasks=$failed ==="; exit 0
  fi
  sleep "$POLL"
done
say "=== watch hit 100h cap; job state: running=$running done=$done_chunks/$NCHUNK ==="
exit 0
