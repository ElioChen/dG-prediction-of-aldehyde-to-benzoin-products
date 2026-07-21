#!/bin/bash
#SBATCH --job-name=b6_arch_scaffold
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/scaffold_disjoint_bde/b6_arch_scaffold_%A_%a.out
#
# Re-check the two architecture-variant wins/non-wins (attentive aggregation, chemprop 2.3.0)
# under the genuinely scaffold-disjoint split instead of just the naive molecule_cold_split
# they were originally tested on -- "all models that need retraining should be rerun"
# (2026-07-17 overnight). Task 0=attentive aldehydes, 1=attentive products,
# 2=chemprop2.3 aldehydes, 3=chemprop2.3 products.
set -o pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV_STABLE="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
ENV_23="/gpfs/scratch1/shared/schen3/envs/bde_gnn_chemprop23"
BDE="$REPO/pipeline/bde"
H="$REPO/data/cross_benzoin/homo_v6"
OUT="$REPO/runs/logs/scaffold_disjoint_bde"
mkdir -p "$OUT"

ID=${SLURM_ARRAY_TASK_ID:-0}
case "$ID" in
  0) PY="$ENV_STABLE/bin/python"; WHICH=aldehydes; EXTRA="--aggregation attentive"; TAG=attentive ;;
  1) PY="$ENV_STABLE/bin/python"; WHICH=products;  EXTRA="--aggregation attentive"; TAG=attentive ;;
  2) PY="$ENV_23/bin/python";     WHICH=aldehydes; EXTRA="";                        TAG=chemprop23 ;;
  3) PY="$ENV_23/bin/python";     WHICH=products;  EXTRA="";                        TAG=chemprop23 ;;
esac
if [ "$WHICH" = "aldehydes" ]; then SPLIT="$H/aldehydes_scaffold_split_from_dG.csv"; else SPLIT="$H/products_scaffold_split.csv"; fi

echo "task $ID: tag=$TAG which=$WHICH node=${SLURMD_NODENAME} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY -u "$BDE/train_gnn_hybrid_bde.py" \
    --which "$WHICH" --depth 4 --message-hidden 500 --eval-on test $EXTRA \
    --split-file "$SPLIT" \
    --out "$OUT/${TAG}_${WHICH}_scaffold_disjoint_result.json"
echo "task $ID done $(date) exit=$?"
