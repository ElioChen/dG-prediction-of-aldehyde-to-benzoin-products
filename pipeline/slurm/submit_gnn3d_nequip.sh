#!/bin/bash
#SBATCH --job-name=gnn3d_nq
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/gnn3d_nq_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/home/schen3/venv/nequip/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GNN3D_TAG=_nequip
time cat /home/schen3/venv/nequip/lib/python3.13/site-packages/torch/lib/*.so > /dev/null 2>&1
echo "3D-GNN(nequip) node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY -u "$REPO/pipeline/analysis/gnn3d_schnet_dimenet.py"
echo "Done $(date) exit=$?"
