#!/bin/bash
#SBATCH --job-name=b6_arch_exp
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/b6_arch_exp_%A_%a.out
#SBATCH --array=0-8
#
# 2026-07-16 evening: three lines of investigation the user asked for in one array.
# All use the winning B6 hyperparams from the earlier 12-config search (depth=4,
# message_hidden=500) except where the whole point is to vary something else.
#
#  0-4: learning curve (aldehydes, full library, --train-frac 0.1/0.25/0.5/0.75/1.0,
#       SAME frozen test set at every fraction via train_gnn_hybrid_bde.py's new
#       --train-frac flag) -- diagnoses data-limited vs architecture-limited.
#  5-6: chemprop 2.3.0 (isolated env envs/bde_gnn_chemprop23, smoke-tested compatible,
#       zero code changes needed) on aldehydes/products at full scale -- does the
#       newer chemprop version change accuracy at all vs the 2.2.4 "b6_best" numbers?
#  7-8: AttentiveAggregation instead of MeanAggregation (aldehydes/products, chemprop
#       2.2.4, standard BondMessagePassing) -- a genuinely different architecture axis,
#       not just a hyperparameter tweak. (MABBondMessagePassing was also tried but
#       chemprop's plain `models.MPNN` wrapper isn't compatible with it -- MAB message
#       passing needs `models.MolAtomBondMPNN`, a bigger integration lift not done here.)
set -o pipefail   # NOT -u/-e: `source /etc/profile` / `module load` reference unset vars and crash under -u

REPO="/scratch-shared/schen3/benzoin-dg"
ENV_STABLE="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
ENV_23="/gpfs/scratch1/shared/schen3/envs/bde_gnn_chemprop23"
BDE="$REPO/pipeline/bde"
OUT="$REPO/runs/logs/b6_arch_exp"
mkdir -p "$OUT"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"

ID=${SLURM_ARRAY_TASK_ID:-0}
COMMON="--depth 4 --message-hidden 500 --eval-on test"

case "$ID" in
  0) PY="$ENV_STABLE/bin/python"; ARGS="--which aldehydes --train-frac 0.1  --out $OUT/lc_ald_010.json --pred-out $OUT/lc_ald_010_pred.csv" ;;
  1) PY="$ENV_STABLE/bin/python"; ARGS="--which aldehydes --train-frac 0.25 --out $OUT/lc_ald_025.json --pred-out $OUT/lc_ald_025_pred.csv" ;;
  2) PY="$ENV_STABLE/bin/python"; ARGS="--which aldehydes --train-frac 0.5  --out $OUT/lc_ald_050.json --pred-out $OUT/lc_ald_050_pred.csv" ;;
  3) PY="$ENV_STABLE/bin/python"; ARGS="--which aldehydes --train-frac 0.75 --out $OUT/lc_ald_075.json --pred-out $OUT/lc_ald_075_pred.csv" ;;
  4) PY="$ENV_STABLE/bin/python"; ARGS="--which aldehydes --train-frac 1.0  --out $OUT/lc_ald_100.json --pred-out $OUT/lc_ald_100_pred.csv" ;;
  5) PY="$ENV_23/bin/python";     ARGS="--which aldehydes --out $OUT/chemprop23_aldehydes.json" ;;
  6) PY="$ENV_23/bin/python";     ARGS="--which products  --out $OUT/chemprop23_products.json" ;;
  7) PY="$ENV_STABLE/bin/python"; ARGS="--which aldehydes --aggregation attentive --out $OUT/attentive_aldehydes.json" ;;
  8) PY="$ENV_STABLE/bin/python"; ARGS="--which products  --aggregation attentive --out $OUT/attentive_products.json" ;;
  *) echo "bad array id $ID"; exit 1 ;;
esac

echo "task $ID: $PY $BDE/train_gnn_hybrid_bde.py $COMMON $ARGS node=${SLURMD_NODENAME} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY "$BDE/train_gnn_hybrid_bde.py" $COMMON $ARGS
echo "task $ID done $(date)"
