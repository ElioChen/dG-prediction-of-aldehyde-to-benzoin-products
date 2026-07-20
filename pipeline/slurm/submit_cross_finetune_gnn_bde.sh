#!/bin/bash
#SBATCH --job-name=xft_gnn
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/xft_gnn_%j.out
#
# B6 (GNN+3D hybrid) homo->cross transfer for product BDE (2026-07-16). The GNN analogue of
# the XGB cross-finetune experiment (PROGRESS_20260714.md 〇-3 sec 5). Pretrains a restricted-
# 36-shared-feature B6 on homo products (~176k rows, the expensive part, checkpointed to
# --pretrain-ckpt for reuse), then reports zero-shot / A cross-only / C fine-tune on the same
# donor-acceptor-cold cross test split. Uses the CSV cross table (the bde_gnn env has no
# pyarrow, can't read the parquet twin).
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
BDE="$REPO/pipeline/bde"
mkdir -p "$REPO/runs/logs"

CROSS_CSV="${CROSS_CSV:-$REPO/data/cross_benzoin/cross_round5/cross_train_table_5rounds_mordred.csv}"
CKPT="${CKPT:-$REPO/runs/logs/b6_homo_products_shared36.pt}"
OUT="${OUT:-$REPO/runs/logs/cross_finetune_gnn_round5.json}"

echo "B6 GNN cross-finetune  cross=$CROSS_CSV  ckpt=$CKPT  $(date)"
$ENV/bin/python -u "$BDE/train_cross_finetune_gnn_bde.py" \
    --cross-table "$CROSS_CSV" --pretrain-ckpt "$CKPT" --out "$OUT"
echo "done $(date) exit=$?"
