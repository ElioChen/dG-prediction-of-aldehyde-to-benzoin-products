#!/bin/bash
#SBATCH --job-name=homo_standalone
#SBATCH --partition=fat_genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/homo_standalone_%j.out
#
# Standalone homo-only dG model: assemble the 30k homo table (champion 260-feat
# schema) then train + eval single-XGB champion and MLP+XGB ensemble on the
# scaffold-disjoint split. GNN leg is a separate dependent job
# (submit_homo_standalone_gnn.sh).
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24

OUT=data/cross_benzoin/homo_standalone
TABLE=$OUT/homo_standalone_train_table_slim260.parquet
mkdir -p "$OUT"
echo "homo_standalone node=${SLURMD_NODENAME} $(date)"

if [[ ! -f "$TABLE" ]]; then
    $PY -u cross_benzoin/assemble_homo_standalone_table.py --out "$TABLE" || exit 1
else
    echo "table exists, skip assemble: $TABLE"
fi

$PY -u cross_benzoin/train_scaffold_disjoint.py \
    --table "$TABLE" --outdir "$OUT/tabular"
RC=$?
echo "Done $(date) exit=$RC"
exit $RC
