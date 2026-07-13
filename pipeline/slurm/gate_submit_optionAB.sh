#!/bin/bash
#SBATCH --job-name=gate_optAB
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/slurm_logs/gate_optAB-%j.out
# SINGLE-FIRE gate: submit Option A/B + 36-hard ONCE, only when inode usage < THRESH
# (low, to leave headroom for the campaign's own scratch). A sentinel guarantees it never
# double-submits even across requeues/cancels. THROTTLED concurrency to bound peak inodes.
REPO="/scratch-shared/schen3/benzoin-dg"
SELF="$REPO/pipeline/slurm/gate_submit_optionAB.sh"
SENTINEL="$REPO/data/raw/screen_v6/dft_sp_r2scan3c/.optAB_submitted"
THRESH=55          # leave ~45% headroom for the campaign's scratch
CAMP_THROTTLE=10   # %10 -> ~10 tasks x 12 workers = ~120 concurrent molecules (inode-safe)

if [[ -f "$SENTINEL" ]]; then echo "sentinel exists — already submitted, gate exiting."; exit 0; fi
pct=$(myquota 2>/dev/null | awk '/wstor_scratch1/{f=1} f&&/Inodes/{gsub("%","",$5);print int($5);exit}')
echo "[$(date '+%F %T')] inode=${pct:-?}% (threshold ${THRESH}%)"
if [[ -z "$pct" ]]; then sbatch --begin=now+15minutes "$SELF"; exit 0; fi
if (( pct < THRESH )); then
  if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qE 'gxtb_solv_camp|dftopt_36hard'; then
    echo "campaign/36hard already queued — gate exiting (no double-submit)"; exit 0
  fi
  cd "$REPO"
  sbatch --array=0-91%${CAMP_THROTTLE} "$REPO/pipeline/slurm/submit_gxtb_solv_campaign.sh"
  sbatch --array=0-35%12 "$REPO/pipeline/slurm/submit_dftopt_36hard.sh"
  date > "$SENTINEL"
  echo "SUBMITTED Option A/B (throttle %${CAMP_THROTTLE}) + 36-hard. sentinel written. gate done — NOT requeueing."
else
  echo "quota ${pct}% >= ${THRESH}% — requeue in 15min"
  sbatch --begin=now+15minutes "$SELF"
fi
