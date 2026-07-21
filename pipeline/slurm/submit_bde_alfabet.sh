#!/bin/bash
#SBATCH --job-name=bde_alfabet
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/alfabet_%x_%A_%a.out
#SBATCH --array=0-1
#
# Full-library ALFABET zero-shot baseline for the aldehyde formyl C-H bond (array idx 0)
# and the product ketC-carbC bond (array idx 1) -- see pipeline/compute/calc_bde_alfabet.py.
# Purely 2D (SMILES-only), CPU-only. First real-cluster attempt (job 24618448) showed the
# fragment-enumeration step alone (not even the TF predict step) takes ~1h40m for the
# 220k aldehyde side and, because product molecules are ~2x the atom/bond count, was only
# 29% done after 1h48m on the product side (~12.6 mol/s vs ~37 mol/s for aldehydes) --
# a 3h budget is nowhere near enough for products. Sized generously at 10h/side now that
# real throughput is known, still run as a 2-way array so both sides proceed in parallel.
#
# ALFABET needs an isolated env (old TF 2.11 + nfp + scikit-learn==0.24.2, py3.9) --
# see pipeline/gnn/README.md for the established dedicated-env pattern this follows.
# Do NOT run this against the main project env or bde_lite/gnn.
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/alfabet"
OUT="$REPO/data/cross_benzoin/homo_v6"
mkdir -p "$OUT"

WHICH=(aldehydes products)
W="${WHICH[$SLURM_ARRAY_TASK_ID]}"

echo "ALFABET full-library BDE baseline ($W) $(date)"
$ENV/bin/python -u "$REPO/pipeline/compute/calc_bde_alfabet.py" \
    --which "$W" --n 999999 --out "$OUT/${W}_bde_alfabet.csv"
echo "$W done $(date) exit=$?"
