#!/bin/bash
#SBATCH --job-name=xft_gnn_r7
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/xft_gnn_r7_%j.out
#
# B6 (GNN+3D hybrid) homo->cross transfer for product BDE, round1-7 scale (2026-07-17),
# using cross-benzoin's own scaffold-disjoint-labeled round1-7 table (train=19,687/
# validation=483/test=450, "mixed" pairs dropped) instead of a fresh donor/acceptor-cold
# split -- see train_cross_finetune_gnn_bde_round7.py docstring. Reuses the SAME cached
# homo pretrain checkpoint as the round5 run (b6_homo_products_shared36.pt) so only the
# cross-side fine-tune/cross-only training is new GPU work. Does not touch or overwrite
# the round5 result (cross_finetune_gnn_round5.json) -- new filenames per this project's
# "preserve output history" convention.
set -o pipefail   # not -u: module load / source reference unset vars

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
BDE="$REPO/pipeline/bde"
mkdir -p "$REPO/runs/logs"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"

CROSS_CSV="${CROSS_CSV:-$REPO/runs/logs/bde_cross_round7_scaffold_split_labeled.csv}"
CKPT="${CKPT:-$REPO/runs/logs/b6_homo_products_shared36.pt}"
OUT="${OUT:-$REPO/runs/logs/cross_finetune_gnn_round7_scaffold.json}"

echo "B6 GNN cross-finetune (round1-7, scaffold-disjoint) cross=$CROSS_CSV ckpt=$CKPT node=${SLURMD_NODENAME} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
"$ENV/bin/python" -u "$BDE/train_cross_finetune_gnn_bde_round7.py" \
    --cross-table "$CROSS_CSV" --pretrain-ckpt "$CKPT" --out "$OUT"
echo "done $(date) exit=$?"
