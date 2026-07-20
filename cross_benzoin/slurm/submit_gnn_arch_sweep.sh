#!/bin/bash
#SBATCH --job-name=gnn_arch_sweep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#
# Architecture + hyperparameter sweep for cross-benzoin's own GNN (run during round8's
# DFT-SP, GPU queue idle, 2026-07-20 user-directed). 6 configs: a baseline-reproduction
# sanity check, the queued AttentiveFP-style pooling comparison, capacity variants
# (wider/deeper), attention+capacity combined, and a lower-lr variant.
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"
TABLE="data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_split_labeled.parquet"
CHAMPION_DIR="data/cross_benzoin/cross_round7/scaffold_disjoint_v1"
ENSEMBLE_PATH="data/cross_benzoin/cross_round7/scaffold_disjoint_v1/models/ensemble_scaffold_disjoint.joblib"
OUTBASE="data/cross_benzoin/cross_round7/gnn_arch_sweep"

# ID: arch hidden layers lr tag
CONFIGS=(
  "default 128 4 3e-4 default_h128_l4_lr3e-4"
  "attentive 128 4 3e-4 attentive_h128_l4_lr3e-4"
  "default 256 4 3e-4 default_h256_l4_lr3e-4"
  "default 128 6 3e-4 default_h128_l6_lr3e-4"
  "attentive 256 4 3e-4 attentive_h256_l4_lr3e-4"
  "default 128 4 1e-4 default_h128_l4_lr1e-4"
)

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
time cat /gpfs/scratch1/shared/schen3/envs/gnn_lite/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1

cd "$REPO"
read -r ARCH HIDDEN LAYERS LR TAG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
OUTDIR="$OUTBASE/$TAG"
mkdir -p "$OUTDIR"
echo "gnn_arch_sweep tag=$TAG node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table "$TABLE" --champion-dir "$CHAMPION_DIR" --ensemble-path "$ENSEMBLE_PATH" \
    --outdir "$OUTDIR" --arch "$ARCH" --hidden "$HIDDEN" --layers "$LAYERS" --lr "$LR" --seed 0
RC=$?
echo "Done $TAG $(date) exit=$RC"
exit "$RC"
