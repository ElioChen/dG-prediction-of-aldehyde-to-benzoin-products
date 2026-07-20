#!/bin/bash
#SBATCH --job-name=bde_cross
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#
# Product-side g-xTB BDE (new ketC-carbC bond) for cross-benzoin products -- the
# descriptor piece missing from the cross Delta-model (donor/acceptor BDE was free,
# reused from the full-library aldehyde cache; product BDE is genuinely new compute,
# one molecule per donor/acceptor pair). See calc_bde_gxtb_product_cross.py.
#
# Submit (PRODUCTS: a cb_featurize.py products.csv, e.g. cross_pilot_v1_products.csv):
#   IN=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/cross_pilot_v1/cross_pilot_v1_products.csv
#   OUT=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/cross_pilot_v1/bde_gxtb
#   N=$(($(wc -l < "$IN")-1)); CHUNK=100; NCH=$(( (N+CHUNK-1)/CHUNK )); mkdir -p "$OUT/logs"
#   sbatch --array=0-$((NCH-1))%48 --output="$OUT/logs/bde_%a.out" \
#     --export=ALL,INPUT="$IN",OUTDIR="$OUT",CHUNK=$CHUNK pipeline/slurm/submit_bde_gxtb_product_cross.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
XTB_BIN="/home/schen3/xtb/bin/xtb"
GXTB_BIN="${GXTB_BIN:-/gpfs/scratch1/shared/schen3/software/g-xtb/linux/xtb-6.7.1/bin/xtb}"
INPUT="${INPUT:?set INPUT=/abs/products.csv}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"
CHUNK="${CHUNK:-100}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ID=$SLURM_ARRAY_TASK_ID
WORK="${TMPDIR:-$REPO/tmp}/bde_cross_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$OUTDIR/logs"
echo "bde_cross ${SLURM_ARRAY_JOB_ID}[$ID] node=${SLURMD_NODENAME} $(date)"
cd "$REPO/pipeline/compute"
$PY -u calc_bde_gxtb_product_cross.py --products-csv "$INPUT" \
    --chunk-id "$ID" --chunk-size "$CHUNK" --out-dir "$OUTDIR" \
    --xtb-bin "$XTB_BIN" --gxtb-bin "$GXTB_BIN" --work-dir "$WORK"
rm -rf "$WORK"
echo "Done task=$ID $(date)"
