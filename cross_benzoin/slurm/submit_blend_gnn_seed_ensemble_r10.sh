#!/bin/bash
#SBATCH --job-name=blend_gnn_seed_r10
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/blend_gnn_seed_r10_%j.out
#
# Inference-only: load the shipped r1-10 champion GNN + gnn_attentive_10rounds_seed{1..4},
# run each on the frozen scaffold-disjoint holdout, average GNN preds over a growing
# number of seeds, re-sweep the blend weight vs the shipped MLP+XGB ensemble.
# Run off the login node (blend_gnn_seed_ensemble_r10.py was SIGTERM'd there).
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "blend_gnn_seed_r10 node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/blend_gnn_seed_ensemble_r10.py
echo "Done $(date) exit=$?"
