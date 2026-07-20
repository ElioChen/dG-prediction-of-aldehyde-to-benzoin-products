#!/bin/bash
#SBATCH --job-name=b6_ckpt
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/scaffold_disjoint_bde/b6_ckpt_%A_%a.out
#
# Re-run of submit_b6_scaffold_disjoint.sh (job 24694779, which produced the confirmed
# champion numbers aldehyde MAE 1.579/R^2 0.843 and product MAE 3.060/R^2 0.886, see
# PROGRESS_20260714.md section O-10) with IDENTICAL hyperparameters and split files, the
# ONLY difference being --save-checkpoint is now passed so a real deployable checkpoint
# gets written (previously enable_checkpointing=False meant no B4/B5/B6 run ever saved
# weights -- see PROGRESS_20260714.md section O-12 "尚未做" and
# predict_bde_champion.py). Does not overwrite the original result/pred files from
# 24694779 -- writes new *_ckpt_* filenames per this project's "preserve output history"
# convention. Task 0 = aldehydes, task 1 = products.
set -o pipefail   # not -u: module load / source reference unset vars

REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/bde_gnn/bin/python"
H="$REPO/data/cross_benzoin/homo_v6"
OUT="$REPO/runs/logs/scaffold_disjoint_bde"
MODELS="$OUT/models"
mkdir -p "$OUT" "$MODELS"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"

ID=${SLURM_ARRAY_TASK_ID:-0}
if [ "$ID" -eq 0 ]; then
  WHICH=aldehydes; SPLIT="$H/aldehydes_scaffold_split_from_dG.csv"
else
  WHICH=products; SPLIT="$H/products_scaffold_split.csv"
fi

echo "scaffold-disjoint retrain (checkpoint-saving) which=$WHICH node=${SLURMD_NODENAME} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY -u pipeline/bde/train_gnn_hybrid_bde.py \
    --which "$WHICH" --depth 4 --message-hidden 500 --eval-on test \
    --split-file "$SPLIT" \
    --pred-out "$OUT/${WHICH}_scaffold_disjoint_ckpt_pred.csv" \
    --out "$OUT/${WHICH}_scaffold_disjoint_ckpt_result.json" \
    --save-checkpoint "$MODELS/b6_${WHICH}_scaffold_disjoint.pt"
echo "done $(date) exit=$?"
