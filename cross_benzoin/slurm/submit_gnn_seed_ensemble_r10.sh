#!/bin/bash
#SBATCH --job-name=gnn_seed_r10
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=1-4
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/gnn_seed_r10_%A_%a.out
#
# Multi-seed attentive-GNN for the r1-10 table. The shipped champion GNN
# (gnn_attentive_10rounds_v1) is a SINGLE seed (seed=0); this trains seeds 1-4 to
# separate dirs so blend_gnn_seed_ensemble_r10.py can test whether seed-averaging
# the GNN leg beats the single-seed blend 2.215.
#
# NOTE (2026-09-08): trained on the CURRENT r10 labels + g-xTB baseline. When the
# Tier B self-consistent relabel campaign (job 26463424) lands, the GNN leg must be
# retrained on the new dG_r2scan + BASELINE_COL=dG_b973c_kcal -- this run's value is
# (a) a seed-variance number now, (b) proving the seed-ensemble recipe before the
# post-campaign retrain.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R10=data/cross_benzoin/cross_round10
SEED=${SLURM_ARRAY_TASK_ID:-1}
OUT=$R10/gnn_attentive_10rounds_seed${SEED}
echo "gnn_seed_r10 seed=$SEED node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
$PY -c "import torch,torch_geometric;print('torch',torch.__version__,'tg',torch_geometric.__version__,'cuda',torch.cuda.is_available())"
$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table $R10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet \
    --champion-dir $R10/scaffold_disjoint_10rounds_v1 \
    --ensemble-path $R10/scaffold_disjoint_10rounds_v1/models/ensemble_scaffold_disjoint.joblib \
    --outdir $OUT \
    --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed $SEED
RC=$?
echo "Done seed=$SEED $(date) exit=$RC"
exit $RC
