#!/bin/bash
#SBATCH --job-name=morfeus_desc
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --time=04:00:00
#
# Compute extra morfeus descriptors (Pyramidalization + atom-resolved Dispersion P_int
# at ketC/carbC for products, CHO_C for aldehydes) for the homo_v6 MAIN library
# (products_all.csv / aldehydes_all.csv, 220,725 / 220,526 rows), reusing the geometry
# already saved (xyz_file) -- no re-optimization. Non-destructive sidecars keyed by `id`:
#   data/cross_benzoin/homo_v6/products_morfeus_extra.csv
#   data/cross_benzoin/homo_v6/aldehydes_morfeus_extra.csv
#
# NOTE: this runs on the main homo_v6 library only. It does NOT cover the ~1300 pairs
# in the in-flight homo_v6_backfill (job 24344127) -- those get morfeus'd separately
# once that backfill lands and is merged in (small addendum, same script).
#
# Submit:
#   cd /gpfs/scratch1/shared/schen3/benzoin-dg
#   sbatch --output=data/cross_benzoin/homo_v6/logs/morfeus_desc_%j.out \
#     cross_benzoin/slurm/submit_morfeus_descriptors.sh
#
REPO="/gpfs/scratch1/shared/schen3/benzoin-dg"
VENV="/home/schen3/venv/nhc-workflow"
OUT="$REPO/data/cross_benzoin/homo_v6"
mkdir -p "$OUT/logs"

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
source "$VENV/bin/activate"

cd "$REPO"
for WHICH in products aldehydes; do
    DEST="$OUT/${WHICH}_morfeus_extra.csv"
    if [[ -s "$DEST" ]]; then
        echo "$WHICH: $DEST already exists — skip"
        continue
    fi
    echo "=== $WHICH  $(date) cpus=$SLURM_CPUS_PER_TASK ==="
    python cross_benzoin/analysis/add_morfeus_descriptors.py "$WHICH"
    echo "=== $WHICH done $(date) ==="
done
echo "ALL DONE $(date)"
