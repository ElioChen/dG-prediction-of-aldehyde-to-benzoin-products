#!/bin/bash
#SBATCH --job-name=homo_prod_pilot
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# Homo-benzoin PRODUCT featurization for screen_v5_pilot: build each aldehyde's
# homo-benzoin product, conformer-search + xTB-opt (SAVES product xyz), compute
# xTB+morfeus+Multiwfn descriptors and xTB ΔG. Single 1-node job over ~2000 pairs.
#
# Submit (homo_pairs.csv is produced from screen_all.csv beforehand):
#   sbatch submit_homo_products_pilot.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
SCRIPT="$REPO/pipeline/compute/featurize_product.py"
PILOT="$REPO/data/raw/screen_v5_pilot"
INPUT="$PILOT/homo_pairs.csv"
OUTDIR="$PILOT/products_homo"
OUTPUT="$OUTDIR/features.csv"      # product xyz go to $OUTDIR/xyz/

VENV="/home/schen3/venv/nhc-workflow"
XTB_BIN="/home/schen3/xtb/bin/xtb"
MWF_BIN="/home/schen3/mutiwfn/Multiwfn_noGUI"
SOLVENT="${SOLVENT:-dmso}"; N_CONFS="${N_CONFS:-10}"
WORKERS=24; XTB_CORES=2           # 24 x 2 = 48 cores

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
source "$VENV/bin/activate"
export XTBPATH="/home/schen3/xtb/share/xtb"
export OMP_NUM_THREADS=$XTB_CORES MKL_NUM_THREADS=$XTB_CORES OMP_STACKSIZE=2G KMP_STACKSIZE=2G

mkdir -p "$OUTDIR/xyz" "$PILOT/logs"
WORK_DIR="${TMPDIR:-$PROJ/tmp}/homo_prod_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT TERM INT

echo "Homo products pilot  node=${SLURMD_NODENAME}  n_confs=$N_CONFS  $(date)"
cd "$REPO/pipeline/compute"
python "$SCRIPT" \
    --input "$INPUT" --output "$OUTPUT" --work-dir "$WORK_DIR" \
    --xtb-bin "$XTB_BIN" --solvent "$SOLVENT" --n-confs "$N_CONFS" \
    --conformer funnel_v3 --workers "$WORKERS" --xtb-cores "$XTB_CORES" --parallel-jobs 1 \
    --multiwfn --multiwfn-bin "$MWF_BIN" \
    2>&1 | tee "$PILOT/logs/homo_products.log"
echo "Done $(date) exit=${PIPESTATUS[0]}  -> $OUTPUT"
