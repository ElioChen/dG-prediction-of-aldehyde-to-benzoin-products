#!/bin/bash
#SBATCH --job-name=rdkit_desc
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --time=04:00:00
#
# Compute the full 217-descriptor RDKit 2D set for the homo_v6 products AND aldehydes
# tables, writing non-destructive sidecars keyed by `id`:
#   data/cross_benzoin/homo_v6/products_rdkit_descriptors.csv
#   data/cross_benzoin/homo_v6/aldehydes_rdkit_descriptors.csv
# Re-submit of the interactive run that got killed. genoa 24-core unit; Pool size reads
# SLURM_CPUS_PER_TASK. Resume-safe: skips a target whose output already exists.
#
# Submit:
#   cd /gpfs/scratch1/shared/schen3/benzoin-dg
#   sbatch --output=data/cross_benzoin/homo_v6/logs/rdkit_desc_%j.out \
#     cross_benzoin/slurm/submit_rdkit_descriptors.sh
#
# NB: no `set -e` — lmod's /etc/profile init fails under `set -e`/`pipefail` (even with
# `|| true`), which is why the first submit (24345818) died in 22s with an empty log. The
# proven featurize submit script sources the env the same plain way.
REPO="/gpfs/scratch1/shared/schen3/benzoin-dg"
VENV="/home/schen3/venv/nhc-workflow"
OUT="$REPO/data/cross_benzoin/homo_v6"
mkdir -p "$OUT/logs"

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
source "$VENV/bin/activate"

cd "$REPO"
for WHICH in products aldehydes; do
    DEST="$OUT/${WHICH}_rdkit_descriptors.csv"
    if [[ -s "$DEST" ]]; then
        echo "$WHICH: $DEST already exists — skip"
        continue
    fi
    echo "=== $WHICH  $(date) cpus=$SLURM_CPUS_PER_TASK ==="
    python cross_benzoin/analysis/add_rdkit_descriptors.py "$WHICH"
    echo "=== $WHICH done $(date) ==="
done
echo "ALL DONE $(date)"
