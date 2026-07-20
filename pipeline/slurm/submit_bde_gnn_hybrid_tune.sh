#!/bin/bash
#SBATCH --job-name=b6_tune
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/b6_tune_%A_%a.out
#SBATCH --array=0-11
#
# B6 hyperparameter search (2026-07-16, PROGRESS_20260714.md 〇-3 todo). 12 curated configs,
# each evaluated with --eval-on val (reports on the early-stopping validation fold, TEST set
# never loaded, so config selection can't leak). aldehydes only (fastest split) as the
# exploratory pass; the winning config is then re-run on BOTH splits with --eval-on test for
# the honest number. Config table below is "depth message_hidden ffn_hidden ffn_layers dropout".
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
BDE="$REPO/pipeline/bde"
OUT="$REPO/runs/logs/b6_tune"
mkdir -p "$OUT"

# depth  message_hidden  ffn_hidden  ffn_layers  dropout
CONFIGS=(
  "4 300 300 2 0.0"   # 0: DEFAULT (reference)
  "5 300 300 2 0.0"   # 1: deeper MP
  "3 300 300 2 0.0"   # 2: shallower MP
  "4 500 300 2 0.0"   # 3: wider MP
  "4 600 300 2 0.0"   # 4: widest MP
  "4 300 500 2 0.0"   # 5: wider FFN
  "4 300 300 3 0.0"   # 6: deeper FFN
  "4 300 300 2 0.1"   # 7: dropout
  "5 500 300 2 0.0"   # 8: deeper+wider MP
  "5 500 500 2 0.1"   # 9: big + regularized
  "6 500 300 2 0.0"   # 10: deepest MP
  "4 500 500 3 0.1"   # 11: wide FFN + deep FFN + dropout
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r DEPTH MH FH FL DO <<< "$CFG"
TAG=$(printf "cfg%02d_d%s_mh%s_fh%s_fl%s_do%s" "$SLURM_ARRAY_TASK_ID" "$DEPTH" "$MH" "$FH" "$FL" "$DO")

echo "B6 tune $TAG (aldehydes, eval-on val) $(date)"
$ENV/bin/python -u "$BDE/train_gnn_hybrid_bde.py" \
    --which aldehydes --eval-on val \
    --depth "$DEPTH" --message-hidden "$MH" --ffn-hidden "$FH" \
    --ffn-layers "$FL" --dropout "$DO" \
    --out "$OUT/${TAG}.json"
echo "$TAG done $(date) exit=$?"
