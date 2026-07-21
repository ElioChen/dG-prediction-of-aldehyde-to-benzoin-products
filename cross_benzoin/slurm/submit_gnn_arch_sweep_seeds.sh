#!/bin/bash
#SBATCH --job-name=gnn_arch_seeds
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#
# Multi-seed confirmation for the two most promising configs from
# submit_gnn_arch_sweep.sh's seed-0 pass (attentive_h128_l4_lr3e-4 looked like a real win,
# but the seed-0 "default" reproduction itself didn't match the shipped champion's own
# reported number -- 2.695 vs 2.523 -- indicating real run-to-run GNN variance that must be
# accounted for before trusting a single-seed comparison, same "always multi-seed before
# promoting" discipline already used elsewhere in this project (B4's 3-seed check, the
# GNN-stacking-null correction).
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"
TABLE="data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_split_labeled.parquet"
CHAMPION_DIR="data/cross_benzoin/cross_round7/scaffold_disjoint_v1"
ENSEMBLE_PATH="data/cross_benzoin/cross_round7/scaffold_disjoint_v1/models/ensemble_scaffold_disjoint.joblib"
OUTBASE="data/cross_benzoin/cross_round7/gnn_arch_sweep"

# ID: arch seed tag
CONFIGS=(
  "default 1 default_h128_l4_lr3e-4_seed1"
  "default 2 default_h128_l4_lr3e-4_seed2"
  "attentive 1 attentive_h128_l4_lr3e-4_seed1"
  "attentive 2 attentive_h128_l4_lr3e-4_seed2"
)

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
time cat /gpfs/scratch1/shared/schen3/envs/gnn_lite/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1

cd "$REPO"
read -r ARCH SEED TAG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
OUTDIR="$OUTBASE/$TAG"
mkdir -p "$OUTDIR"
echo "gnn_arch_sweep_seeds tag=$TAG node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table "$TABLE" --champion-dir "$CHAMPION_DIR" --ensemble-path "$ENSEMBLE_PATH" \
    --outdir "$OUTDIR" --arch "$ARCH" --hidden 128 --layers 4 --lr 3e-4 --seed "$SEED"
RC=$?
echo "Done $TAG $(date) exit=$RC"
exit "$RC"
