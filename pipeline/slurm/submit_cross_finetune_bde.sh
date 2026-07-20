#!/bin/bash
#SBATCH --job-name=cross_ft_bde
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/cross_ft_bde_%j.out
#
# BDE Phase 1 next-step #2 (pipeline/bde/train_cross_finetune_bde.py): cross-only vs
# homo+cross joint vs homo-pretrain+cross-finetune, all scored on the same held-out cross
# product BDE test split, plus a matched zero-shot baseline. Loads the full 208k-row homo
# product table (same class of job that killed a login-node run for B3-217 -- see
# submit_dspoc_baseline.sh), so run on a compute node.
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_lite"
mkdir -p "$REPO/runs/logs"

OUT="${OUT:-$REPO/runs/logs/cross_finetune_bde_result.json}"

echo "cross fine-tune/joint BDE experiment $(date)"
$ENV/bin/python -u "$REPO/pipeline/bde/train_cross_finetune_bde.py" --out "$OUT"
echo "done $(date) exit=$?"
