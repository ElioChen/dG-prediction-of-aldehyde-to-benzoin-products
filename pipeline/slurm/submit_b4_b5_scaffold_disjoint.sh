#!/bin/bash
#SBATCH --job-name=b4b5_scaffold
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/scaffold_disjoint_bde/b4b5_scaffold_%A_%a.out
#
# Extend the scaffold-disjoint honest re-evaluation (already done for B6, job 24694779,
# confirmed real +43%/+47% degradation) to B4 (D-MPNN) and B5 (BonDNet-style) -- the user
# asked "all models that need retraining should be rerun" under the new scaffold-disjoint
# split. Task 0=B4 aldehydes, 1=B4 products, 2=B5 aldehydes, 3=B5 products.
set -o pipefail   # not -u/-e: safe here (no module load/etc-profile calls), but kept
                   # consistent with the rest of this project's convention

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
BDE="$REPO/pipeline/bde"
H="$REPO/data/cross_benzoin/homo_v6"
OUT="$REPO/runs/logs/scaffold_disjoint_bde"
mkdir -p "$OUT"

ID=${SLURM_ARRAY_TASK_ID:-0}
case "$ID" in
  0) MODEL=b4; WHICH=aldehydes ;;
  1) MODEL=b4; WHICH=products ;;
  2) MODEL=b5; WHICH=aldehydes ;;
  3) MODEL=b5; WHICH=products ;;
esac
SPLIT="$H/${WHICH}_scaffold_split_from_dG.csv"
[ "$WHICH" = "aldehydes" ] && SPLIT="$H/aldehydes_scaffold_split_from_dG.csv" || SPLIT="$H/products_scaffold_split.csv"

echo "task $ID: model=$MODEL which=$WHICH node=${SLURMD_NODENAME} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

if [ "$MODEL" = "b4" ]; then
  $ENV/bin/python -u "$BDE/train_gnn_bde.py" \
      --which "$WHICH" --split-file "$SPLIT" \
      --out "$OUT/b4_${WHICH}_scaffold_disjoint_result.json" \
      --pred-out "$OUT/b4_${WHICH}_scaffold_disjoint_pred.csv"
else
  ALFABET_CSV="$REPO/data/cross_benzoin/homo_v6/${WHICH}_bde_alfabet.csv"
  $ENV/bin/python -u "$BDE/train_bondnet_bde.py" \
      --which "$WHICH" --alfabet-csv "$ALFABET_CSV" --max-epochs 40 \
      --split-file "$SPLIT" \
      --out "$OUT/b5_${WHICH}_scaffold_disjoint_result.json" \
      --pred-out "$OUT/b5_${WHICH}_scaffold_disjoint_pred.csv"
fi
echo "task $ID done $(date) exit=$?"
