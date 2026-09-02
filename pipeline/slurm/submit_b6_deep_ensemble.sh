#!/bin/bash
#SBATCH --job-name=b6_ensemble
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/runs/logs/scaffold_disjoint_bde/b6_ensemble_%A_%a.out
#
# B6 champion (D-MPNN + H-SPOC local-descriptor x_d fusion, depth=4 message_hidden=500,
# the winning hyperparams) as a 5-seed DEEP ENSEMBLE per task, scaffold-disjoint split.
#
# WHY: STATUS.md sections 三/六 and the g-xTB-baseline-failure finding all point to the
# same next step -- an inference-time "route hard cases to DFT" rule. That needs a
# calibrated epistemic-uncertainty signal per prediction. A single B6 checkpoint gives
# none; a deep ensemble's per-molecule prediction std is the cheapest well-behaved one.
# Also yields a small MAE gain from ensemble averaging (free, no new compute beyond the
# 4 extra trainings).
#
# array id -> (task, seed):  task = id / 5   (0 aldehydes, 1 products);  seed = id % 5
# Needs data/cross_benzoin/homo_v6/{aldehydes_all,products_all}.csv at FULL size, i.e.
# AFTER featurize array 26316404 + assemble_homo_descriptor_libs.py. Run
# aggregate_b6_ensemble.py afterwards to build the mean/std prediction table.
set -o pipefail   # not -u: module load / source reference unset vars

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/bde_gnn/bin/python}"
H="$REPO/data/cross_benzoin/homo_v6"
export BDE_HOMO_V6="$H"
OUT="$REPO/runs/logs/scaffold_disjoint_bde/ensemble"
MODELS="$OUT/models"
mkdir -p "$OUT" "$MODELS"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"

ID=${SLURM_ARRAY_TASK_ID:-0}
TASK=$(( ID / 5 ))
SEED=$(( ID % 5 ))
if [ "$TASK" -eq 0 ]; then
  WHICH=aldehydes; SPLIT="$H/aldehydes_scaffold_split_from_dG.csv"
else
  WHICH=products;  SPLIT="$H/products_scaffold_split.csv"
fi

echo "B6 deep-ensemble which=$WHICH seed=$SEED node=${SLURMD_NODENAME} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY -u pipeline/bde/train_gnn_hybrid_bde.py \
    --which "$WHICH" --depth 4 --message-hidden 500 --eval-on test --seed "$SEED" \
    --split-file "$SPLIT" \
    --pred-out "$OUT/${WHICH}_seed${SEED}_pred.csv" \
    --out "$OUT/${WHICH}_seed${SEED}_result.json" \
    --save-checkpoint "$MODELS/b6_${WHICH}_scaffold_disjoint_seed${SEED}.pt"
echo "done $(date) exit=$?"
