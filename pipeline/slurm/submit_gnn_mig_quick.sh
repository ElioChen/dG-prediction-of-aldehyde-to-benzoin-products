#!/bin/bash
#SBATCH --job-name=gnn_mig
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/gnn_mig_%j.out
#
# QUICK single-GINE Δ-GNN on a gpu_mig A100 20GB slice (easiest to schedule -> earliest result).
# Outputs use SUFFIX=_migquick so they don't clobber the H100 multi-arch sweep.
#   sbatch pipeline/slurm/submit_gnn_mig_quick.sh
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="$PROJ/envs/gnn/bin/python"
SCRIPT="$REPO/pipeline/analysis/gnn_delta_gxtb_dft.py"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# prewarm libtorch into page cache (avoid cold-GPFS import stall; see memory gnn-env-cold-import-slow)
time cat "$PROJ"/envs/gnn/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1
export SUFFIX="_migquick" MAXEP=70
echo "GNN MIG quick node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
$PY -u "$SCRIPT"
echo "Done $(date) exit=$?"
