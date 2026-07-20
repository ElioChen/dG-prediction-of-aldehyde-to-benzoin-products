#!/bin/bash
#SBATCH --job-name=b6_scaffold
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/scaffold_disjoint_bde/b6_scaffold_%A_%a.out
#
# B6 champion (depth=4, message_hidden=500, the winning hyperparams from the 12-config
# search) retrained + honestly evaluated under a GENUINELY scaffold-disjoint split
# (2026-07-17 overnight, coordinated with the neighboring cross-benzoin conversation's own
# scaffold-disjoint rebuild to avoid duplicating the aldehyde-side work -- see
# pipeline/bde/build_scaffold_splits.py docstring). Task 0 = aldehydes, task 1 = products.
set -o pipefail   # not -u: module load / source reference unset vars

REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/bde_gnn/bin/python"
H="$REPO/data/cross_benzoin/homo_v6"
OUT="$REPO/runs/logs/scaffold_disjoint_bde"
mkdir -p "$OUT"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"

ID=${SLURM_ARRAY_TASK_ID:-0}
if [ "$ID" -eq 0 ]; then
  WHICH=aldehydes; SPLIT="$H/aldehydes_scaffold_split_from_dG.csv"
else
  WHICH=products; SPLIT="$H/products_scaffold_split.csv"
fi

echo "scaffold-disjoint retrain which=$WHICH node=${SLURMD_NODENAME} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY -u pipeline/bde/train_gnn_hybrid_bde.py \
    --which "$WHICH" --depth 4 --message-hidden 500 --eval-on test \
    --split-file "$SPLIT" \
    --pred-out "$OUT/${WHICH}_scaffold_disjoint_pred.csv" \
    --out "$OUT/${WHICH}_scaffold_disjoint_result.json"
echo "done $(date) exit=$?"
