#!/bin/bash
#SBATCH --job-name=gnn3d_cmp
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# DGT-inspired 3D-geometry ablation (see gnn3d_train_and_compare.py, gnn_architectures.py's
# TripleGNN3D). Run twice, once per ARCH, as two separate sbatch submissions for genuine
# independence (same convention as submit_gnn_attentive_9rounds.sh's arch sweep):
#   sbatch --export=ALL,ARCH=attentive2d,OUTDIR=data/cross_benzoin/gnn3d/attentive2d_matched_v1 \
#     cross_benzoin/slurm/submit_gnn3d_compare.sh
#   sbatch --export=ALL,ARCH=attentive3d,OUTDIR=data/cross_benzoin/gnn3d/attentive3d_v1 \
#     cross_benzoin/slurm/submit_gnn3d_compare.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"
ARCH="${ARCH:?set ARCH=attentive2d or attentive3d}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/or/relative/out/dir}"
TABLE="${TABLE:-data/cross_benzoin/cross_round9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet}"
BONDLEN="${BONDLEN:-data/cross_benzoin/gnn3d/product_bond_lengths_r2348_r9.parquet}"
CHAMPDIR="${CHAMPDIR:-data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1}"
ENSPATH="${ENSPATH:-data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1/models/ensemble_scaffold_disjoint.joblib}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
time cat /gpfs/scratch1/shared/schen3/envs/gnn_lite/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1

cd "$REPO"
echo "gnn3d_compare arch=$ARCH node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
$PY -u cross_benzoin/gnn3d_train_and_compare.py \
    --table "$TABLE" --bond-lengths "$BONDLEN" \
    --champion-dir "$CHAMPDIR" --ensemble-path "$ENSPATH" \
    --outdir "$OUTDIR" \
    --arch "$ARCH" --hidden 128 --layers 4 --lr 3e-4 --seed 0
RC=$?
echo "Done $ARCH $(date) exit=$RC"
exit "$RC"
