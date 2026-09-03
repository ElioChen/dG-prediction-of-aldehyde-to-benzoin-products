#!/bin/bash
#SBATCH --job-name=assemble_r10feat
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Assemble Round10 stage1's unlabeled candidate feature table (no DFT join) so
# score_round_active_learning.py can rank it by ensemble uncertainty. Uses the
# rebuilt aldehyde library (aldehydes_all.csv / aldehydes_bdfe_gxtb_descriptors.csv /
# aldehydes_mordred_slim102.csv) and Round10's own product bde_gxtb/mordred_products.
#
REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/nhc-workflow/bin/python}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO" || exit 1
echo "assemble_round10_fat_features node=${SLURMD_NODENAME:-unknown} $(date)"
$PY -u cross_benzoin/assemble_cross_round_features.py \
    --tag cross_round10_fat20_stage1 \
    --products-csv data/cross_benzoin/cross_round10_fat20_stage1/cross_round10_products_merged.csv
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
