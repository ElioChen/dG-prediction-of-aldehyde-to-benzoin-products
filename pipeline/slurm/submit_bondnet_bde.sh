#!/bin/bash
#SBATCH --job-name=bondnet_bde
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/bondnet_bde_%x_%j.out
#
# Phase-1 baseline B5 (BonDNet-style reaction-difference D-MPNN, pipeline/bde/
# train_bondnet_bde.py) at full library scale. Smoke-tested at n=1932 (R2=0.51); this
# runs the real ~213k-row aldehyde set. No GPU on the interactive/login node this repo's
# work usually runs from, so -- same as submit_bde_gnn.sh (B4) -- this needs the A100.
#
# WHICH is passed in via --export=ALL,WHICH=aldehydes|products (default aldehydes).
# Uses the same isolated envs/bde_gnn as B4 (NOT envs/gnn, which was found corrupted).
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_gnn"
BDE="$REPO/pipeline/bde"
mkdir -p "$REPO/runs/logs"

WHICH="${WHICH:-aldehydes}"
ALFABET_CSV="$REPO/data/cross_benzoin/homo_v6/${WHICH}_bde_alfabet.csv"

echo "BonDNet-style BDE baseline ($WHICH) $(date)"
$ENV/bin/python -u "$BDE/train_bondnet_bde.py" \
    --which "$WHICH" --alfabet-csv "$ALFABET_CSV" --max-epochs 40 \
    --out "$REPO/runs/logs/bondnet_bde_${WHICH}_result.json" \
    --pred-out "$REPO/runs/logs/bondnet_bde_${WHICH}_test_predictions.csv"
echo "$WHICH done $(date) exit=$?"
