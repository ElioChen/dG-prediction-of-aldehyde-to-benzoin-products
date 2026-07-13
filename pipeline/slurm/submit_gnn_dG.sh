#!/bin/bash
#SBATCH --job-name=gnn_dG
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#
# Train a PyG GINE GNN to predict xTB benzoin ΔG from 2D structure (aromatic set),
# scaffold-split, to test whether a graph representation beats the ~0.66 scalar
# descriptor ceiling. Single A100.
#   sbatch submit_gnn_dG.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="$PROJ/envs/gnn/bin/python"        # project GNN env (torch2.12/cu130 + PyG2.8 + rdkit); was nequip which lacked rdkit → the 06-18 crash
SCRIPT="$REPO/data/raw/screen_v6/gnn_dG.py"

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "GNN dG  node=${SLURMD_NODENAME}  gpu=${CUDA_VISIBLE_DEVICES}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
cd "$REPO/data/raw/screen_v6"
$PY -u "$SCRIPT"
echo "Done $(date) exit=$?"
