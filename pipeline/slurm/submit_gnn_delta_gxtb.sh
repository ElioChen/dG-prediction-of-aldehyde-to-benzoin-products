#!/bin/bash
#SBATCH --job-name=gnn_dxtb
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/gnn_delta_%j.out
#
# GINE Δ-GNN: predict (DFT − g-xTB) from product SMILES, n≈136k DFT labels (in-progress job).
# Head-to-head vs GBT Δ (test MAE 2.46). Then predict correction for the whole library.
#   sbatch pipeline/slurm/submit_gnn_delta_gxtb.sh
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="$PROJ/envs/gnn/bin/python"        # torch + PyG2.8 + rdkit
SCRIPT="$REPO/pipeline/analysis/gnn_delta_gxtb_dft.py"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# prewarm libtorch into page cache (avoid cold-GPFS import stall; see memory gnn-env-cold-import-slow)
time cat "$PROJ"/envs/gnn/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1
echo "GNN Δ-gxtb node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
$PY -u "$SCRIPT"
echo "Done $(date) exit=$?"
