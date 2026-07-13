#!/bin/bash
#SBATCH --job-name=bf_mwf_pilot
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# Backfill aldehyde Multiwfn (ADCH/QTAIM) descriptors onto screen_v5_pilot, reusing
# the geometries already saved in screen_all.csv (xyz_file) — NO re-optimization.
# Single 1-node job over all ~2000 pilot aldehydes (avoids the v6 array throttle).
#
# Submit:
#   sbatch submit_backfill_mwf_pilot.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
SCRIPT="$REPO/pipeline/compute/backfill_multiwfn.py"
PILOT="$REPO/data/raw/screen_v5_pilot"
INPUT="$PILOT/screen_all.csv"
OUTPUT="$PILOT/ald_multiwfn.csv"

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"
MWF_BIN="/home/schen3/mutiwfn/Multiwfn_noGUI"
WORKERS=24                       # 24 workers x 2 xtb threads = 48 cores

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
source "$VENV/bin/activate"
export XTBPATH="/home/schen3/xtb/share/xtb"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_STACKSIZE=2G KMP_STACKSIZE=2G

WORK_DIR="${TMPDIR:-$PROJ/tmp}/bf_mwf_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT TERM INT

echo "Backfill MWF pilot  node=${SLURMD_NODENAME}  $(date)"
cd "$REPO/pipeline/compute"
python "$SCRIPT" \
    --input "$INPUT" --output "$OUTPUT" --work-dir "$WORK_DIR" \
    --xtb-bin "$XTB_BIN" --multiwfn-bin "$MWF_BIN" --workers "$WORKERS" \
    2>&1 | tee "$PILOT/logs/backfill_mwf.log"
echo "Done $(date) exit=${PIPESTATUS[0]}  -> $OUTPUT"
