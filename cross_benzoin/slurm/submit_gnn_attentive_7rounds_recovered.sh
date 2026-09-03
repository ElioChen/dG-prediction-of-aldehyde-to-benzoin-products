#!/bin/bash
#SBATCH --job-name=gnn_attn_7r_rec
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/gnn_attn_7r_rec_%j.out
#
# Retrain the attentive-pooling blend GNN on the RECOVERED rounds-1-7 table -- all
# cross-benzoin GNN .pt weights were lost in the 2026-07 purge
# (RECOVERY_REPORT_20260902 section 2). Restores a complete rounds-1-7 champion
# (tabular ensemble MAE 1.877 + GNN blend). r1-9 / r1-10 GNNs follow once those
# DFT labels land.
#
# Env: the old envs/gnn_lite is purged; /home/schen3/venv/nequip has torch 2.11 +
# torch_geometric 2.7 (GINEConv / GlobalAttention / to_dense_batch). Table carries
# new_scaffold_split from relabel_scaffold_split_7rounds_recovered.py.
#
set -o pipefail
REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/nequip/bin/python}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "gnn_attentive_7rounds_recovered node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} py=$PY $(date)"
$PY -c "import torch,torch_geometric;print('torch',torch.__version__,'tg',torch_geometric.__version__,'cuda',torch.cuda.is_available())"

$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table data/cross_benzoin/cross_round7/cross_train_table_7rounds_recovered_scaffold_split_labeled_slim260.parquet \
    --champion-dir data/cross_benzoin/cross_round7/train_ensemble_7rounds_recovered_v1 \
    --ensemble-path data/cross_benzoin/cross_round7/train_ensemble_7rounds_recovered_v1/models/cross_ensemble_model.joblib \
    --outdir data/cross_benzoin/cross_round7/gnn_attentive_7rounds_recovered_v1 \
    --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed 0
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
