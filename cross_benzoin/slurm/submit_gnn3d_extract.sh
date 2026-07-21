#!/bin/bash
#SBATCH --job-name=gnn3d_extract
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#
# Product-side real 3D bond-length extraction (see gnn3d_extract_product_geometry.py's
# module docstring). Single-core, glob + RDKit parsing over ~40k rows across 5 rounds --
# a few minutes' worth of real CPU work, so it goes through sbatch rather than running
# inline on the login node (see memory no-login-node-compute.md).
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
TABLE="${TABLE:?set TABLE=/abs/path/to/table.parquet}"
ROUNDS="${ROUNDS:?set ROUNDS=\"round2 round3 round4 round8 round9\"}"
OUT="${OUT:?set OUT=/abs/path/to/output.parquet}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "gnn3d_extract node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/gnn3d_extract_product_geometry.py \
    --table "$TABLE" --rounds $ROUNDS --out "$OUT"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
