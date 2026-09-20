#!/bin/bash
#SBATCH --job-name=homo_chemprop_full
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/homo_chemprop_full_%j.out
#
# 2026-09-20 (user: benchmark mainstream models): Chemprop (D-MPNN,
# MulticomponentMPNN over product+donor+acceptor) on the homo full-library
# table, mirroring submit_cross_gnn_chemprop.sh's cross-side run. Chemprop is
# a standard, widely-used molecular-property GNN framework distinct from this
# project's own TripleGNN/attentive implementations -- the most "mainstream"
# GNN comparison point available. Smoke-tested first (--smoke, 300-row CPU
# subset, clean).
set -euo pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="/home/schen3/venv/bde_gnn/bin/python"
CB="$REPO/cross_benzoin"
mkdir -p "$REPO/slurm_logs"

# homo table's true label is dG_orca_kcal (renamed from dG_r2scan_kcal by
# assemble_homo_standalone_table.py), NOT dG_r2scan_kcal like the cross table
# -- do not copy the cross submit script's CB_TARGET_COL value here.
export CB_TARGET_COL=dG_orca_kcal
export CB_BASELINE_COL=dG_b973c_kcal

SEED="${SEED:-0}"
OUT="$REPO/data/cross_benzoin/homo_standalone/gnn_chemprop_full_seed${SEED}"

echo "homo chemprop GNN seed=$SEED $(date)"
$PY -u "$CB/train_cross_gnn_chemprop.py" \
    --table "$REPO/data/cross_benzoin/homo_standalone/homo_standalone_full_library_table_slim260.parquet" \
    --feature-list "$REPO/data/cross_benzoin/feature_list_257_no_nCHO_v2.json" \
    --outdir "$OUT" \
    --depth 4 --hidden 300 --ffn-hidden 300 --ffn-layers 2 --dropout 0.0 \
    --batch-size 64 --max-epochs 150 --patience 15 --seed "$SEED"
echo "done $(date) exit=$?"
