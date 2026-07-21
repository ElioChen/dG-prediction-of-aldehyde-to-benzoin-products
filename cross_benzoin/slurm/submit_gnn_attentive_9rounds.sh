#!/bin/bash
#SBATCH --job-name=gnn_attn_9r
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Attentive-pooling GNN (confirmed 3-seed winner, see gnn_arch_sweep) retrained on the
# full round1-9 scaffold-disjoint (80/10/10 production) data, to become the new
# candidate champion alongside the round1-9 champion+ensemble retrain.
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
time cat /gpfs/scratch1/shared/schen3/envs/gnn_lite/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1

cd "$REPO"
echo "gnn_attentive_9rounds node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table data/cross_benzoin/cross_round9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet \
    --champion-dir data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1 \
    --ensemble-path data/cross_benzoin/cross_round9/scaffold_disjoint_9rounds_v1/models/ensemble_scaffold_disjoint.joblib \
    --outdir data/cross_benzoin/cross_round9/gnn_attentive_9rounds_v1 \
    --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed 0
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
