#!/bin/bash
#SBATCH --job-name=assemble_v3
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "assemble_training_table_v3 node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/assemble_cross_training_table_v3.py --rounds 1 2 3 4 5 6 7 8 \
    --product-mordred-csv \
    data/cross_benzoin/cross_round3/rounds123_products_mordred.csv \
    data/cross_benzoin/cross_round4/round4_products_mordred.csv \
    data/cross_benzoin/screen10k/screen10k_products_mordred.csv \
    data/cross_benzoin/cross_round8/round8_products_mordred.csv
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
