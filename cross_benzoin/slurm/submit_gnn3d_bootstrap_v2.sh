#!/bin/bash
#SBATCH --job-name=gnn3d_boot_v2
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

cd "$REPO"
echo "gnn3d_bootstrap_v2 node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/gnn3d_bootstrap_compare_v2.py \
    --table data/cross_benzoin/cross_round9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet \
    --coords data/cross_benzoin/gnn3d/product_coords_r2348_r9.parquet \
    --champion-dir data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1 \
    --state-2d data/cross_benzoin/gnn3d/attentive2d_v2_matched/models/gnn_state.pt \
    --state-dist data/cross_benzoin/gnn3d/distattn_v1/models/gnn_state.pt \
    --seed 0 --n-boot 20000
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
