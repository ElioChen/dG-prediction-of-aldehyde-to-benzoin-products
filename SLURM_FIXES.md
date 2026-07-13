# SLURM Job Cleanup Race Condition Fixes

**Commit**: d15b1bb (Fix slurm job cleanup race conditions and disk quota handling)

## Problem
Task 24123145 and similar SLURM array jobs were experiencing failures due to:

1. **False Orphan Detection**: The `clean_node_orphans.sh` script had `GRACE_MIN=30` (minutes),
   causing it to incorrectly identify tasks that took >30 min as "dead jobs" and delete their
   scratch directories **while they were still running**. This particularly affected array jobs
   with concurrent execution under high load.

2. **SLURM Controller Throttling**: Under high array-job load, `scontrol show job` responses
   could be partial or throttled, causing the script to mistake unresponded job IDs as "dead"
   and delete their scratch directories mid-run.

3. **Disk Quota Errors**: When `/scratch-local` filled up, `featurize_product.py` would crash
   with `OSError: [Errno 122] Disk quota exceeded` without graceful error handling or recovery guidance.

## Solutions Applied

### 1. `pipeline/slurm/clean_node_orphans.sh`
- **GRACE_MIN**: Increased from **30 minutes → 360 minutes (6 hours)** to match SLURM job timeout.
  This ensures no scratch dir is touched unless it has not been modified for 6 hours.
- **is_dead() timeout**: Added `timeout 5 scontrol show job` with safe fallback.
  If SLURM is slow/unreachable, assume the job is alive (never delete when uncertain).

### 2. `pipeline/slurm/submit_product_bvalid.sh`
- **Cleanup wrapper**: Added `timeout 30` around the cleanup call to prevent blocking
  on slow SLURM controller during concurrent job starts.

### 3. `pipeline/compute/featurize_product.py`
- **Disk quota detection**: Added explicit `OSError` handling at file open time.
  - Detect errno 122 (ENOSPC/Disk quota exceeded)
  - Provide clear error message with recovery guidance
  - Preserve `work_dir` location in error log for investigation
  - Also catch quota errors during CSV writing
  - Exit with `return 1` to fail gracefully instead of crashing mid-task

## Impact
These fixes prevent:
- Live SLURM tasks from having their scratch directories deleted mid-execution
- Cascade failures in array jobs due to false orphan detection
- Cryptic "Disk quota exceeded" crashes in featurize scripts
- Blocking/hanging when SLURM controller is slow or throttled under load

## Testing Recommendations
1. Monitor slurm_logs for the cleanup script's timeout handling
2. Re-submit task 24123145 with current fixes
3. Watch for successful completion or graceful disk-quota errors with recovery guidance
4. Verify no scratch directories are deleted during running jobs with `ls -la /scratch-local/`

## Related Configs
- **Job timeouts**: submit_product_bvalid.sh `--time=06:00:00` (6 hours) → must be ≥ GRACE_MIN
- **Array throttling**: `--array=0-33%64` (max 64 concurrent tasks per array)
- **Cleanup cadence**: orphan_cleanup_cron.sh runs every 3 hours (ORPHAN_CADENCE)
