#!/bin/bash
#SBATCH --job-name=cross_gnn_crg_sweep
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/cross_gnn_crg_sweep_%A.out
#
# Hyperparameter sweep arm for the CRG-GNN comparison point (user-requested
# 2026-09-16, following the 30-seed default-hyperparameter result: mean MAE
# 0.572 +/- 0.012, 4-seed-ensemble plateau 0.543, vs champion blend 0.5295 --
# CRG used TripleGNN's un-tuned defaults, so this sweeps its own hyperparams
# to see how much of that ~0.014 gap is closeable). Selection is on VALIDATION
# MAE (not test) to avoid test-set leakage in model selection; a short-list of
# the best configs then gets re-evaluated with seed-ensembling on test.
set -euo pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="/home/schen3/venv/nequip/bin/python"
CB="$REPO/cross_benzoin"
mkdir -p "$REPO/slurm_logs"

export CB_TARGET_COL=dG_r2scan_kcal
export CB_BASELINE_COL=dG_b973c_kcal

TAG="${TAG:?set TAG=<config-name>}"
HIDDEN="${HIDDEN:-128}"
LAYERS="${LAYERS:-4}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.1}"
WD="${WD:-1e-5}"
SEED="${SEED:-0}"
OUT="$REPO/data/cross_benzoin/cross_round10/gnn_crg_sweep/${TAG}_seed${SEED}"

echo "CRG sweep tag=$TAG hidden=$HIDDEN layers=$LAYERS lr=$LR dropout=$DROPOUT wd=$WD seed=$SEED $(date)"
$PY -u "$CB/train_cross_gnn_crg.py" \
    --table "$REPO/data/cross_benzoin/cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet" \
    --feature-list "$REPO/data/cross_benzoin/feature_list_257_no_nCHO_v2.json" \
    --outdir "$OUT" \
    --hidden "$HIDDEN" --layers "$LAYERS" --batch-size 64 --max-epochs 150 --patience 15 \
    --lr "$LR" --dropout "$DROPOUT" --weight-decay "$WD" --seed "$SEED"
echo "done $(date) exit=$?"
