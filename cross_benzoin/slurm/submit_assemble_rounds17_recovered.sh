#!/bin/bash
#SBATCH --job-name=assemble_r17
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Reassemble the rounds 1-7 training table from post-purge survivors, as the
# reproduction test for the whole recovery (target: the surviving 7-round run's
# CV MAE 1.8830 / RMSE 3.7520 / R2 0.5629 over 32,456 rows x 260 features).
#
# Rounds 8-9 are deliberately excluded: their DFT labels have no surviving copy.
#
# Three things differ from how this ran originally, all of them substitutions of
# equal-provenance data rather than approximations:
#   - labels come from data/raw/dft_sp_cross/, reconstructed by
#     synthesize_dft_sp_from_cv_predictions.py out of the tracked cv_predictions dumps
#   - round5's own bde_gxtb/ was lost; its products came from screen10k, whose chunks
#     survive and cover all 4,892 round5 rows (copied into cross_round5/bde_gxtb/)
#   - rounds 5-7 product mordred likewise comes from screen10k_products_mordred.csv,
#     re-merged from the surviving 204 mordred_products_screen10k chunks
#
REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/nhc-workflow/bin/python}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO" || exit 1
echo "assemble_rounds17_recovered node=${SLURMD_NODENAME:-unknown} $(date)"
$PY -u cross_benzoin/assemble_cross_training_table_v3.py --rounds 1 2 3 4 5 6 7 \
    --out-tag 7rounds_recovered \
    --product-mordred-csv \
    data/cross_benzoin/cross_round3/rounds123_products_mordred.csv \
    data/cross_benzoin/cross_round4/round4_products_mordred.csv \
    data/cross_benzoin/screen10k/screen10k_products_mordred.csv
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
