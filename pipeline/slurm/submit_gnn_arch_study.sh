#!/bin/bash
#SBATCH --job-name=gnn_arch
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/gnn_arch_%j.out
#
# Multi-architecture GNN study (GINE/GINE-big/GAT/GCN/NNConv/GINE-hybrid) Δ-learning
# (DFT − g-xTB) from product SMILES, n≈136k. On H100 (faster + freer than a100 right now).
#   sbatch pipeline/slurm/submit_gnn_arch_study.sh
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="$PROJ/envs/gnn/bin/python"        # torch + PyG2.8 + rdkit
SCRIPT="$REPO/pipeline/analysis/gnn_arch_study_gxtb_dft.py"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
# limit BLAS threads (GPU does the math; high counts slow import-time thread-pool spin-up)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# prewarm the 1.2GB libtorch .so into page cache to avoid the ~20min cold-GPFS import stall
# (see memory gnn-env-cold-import-slow)
time cat "$PROJ"/envs/gnn/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1
echo "GNN arch study node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
$PY -u "$SCRIPT"
echo "Done $(date) exit=$?"
