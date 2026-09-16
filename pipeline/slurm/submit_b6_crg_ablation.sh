#!/bin/bash
#SBATCH --job-name=b6_crg_ablation
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/b6_crg_ablation_%A.out
set -o pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/bde_gnn/bin/python}"
H="$REPO/data/cross_benzoin/homo_v6"
export BDE_HOMO_V6="$H"
OUT="${OUT:-$REPO/runs/logs/scaffold_disjoint_bde/crg_ablation}"
mkdir -p "$OUT"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"

SEED="${SEED:?set SEED=<int>}"
MARK="${MARK:?set MARK=0 or 1}"
TAG=$([ "$MARK" = "1" ] && echo marked || echo unmarked)
ARGS=""
[ "$MARK" = "0" ] && ARGS="--no-mark"

echo "B6 CRG ablation tag=$TAG seed=$SEED node=${SLURMD_NODENAME} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY -u pipeline/bde/train_gnn_hybrid_bde_crg.py \
    --depth 4 --message-hidden 500 --eval-on test --seed "$SEED" $ARGS \
    --split-file "$H/products_scaffold_split.csv" \
    --pred-out "$OUT/${TAG}_seed${SEED}_pred.csv" \
    --out "$OUT/${TAG}_seed${SEED}_result.json"
echo "done $(date) exit=$?"
