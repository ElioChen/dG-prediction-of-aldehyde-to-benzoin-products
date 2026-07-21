#!/bin/bash
#SBATCH --job-name=gnn3d_coords
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
TABLE="${TABLE:?set TABLE}"
ROUNDS="${ROUNDS:?set ROUNDS}"
OUT="${OUT:?set OUT}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "gnn3d_extract_coords node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/gnn3d_extract_product_coords.py --table "$TABLE" --rounds $ROUNDS --out "$OUT"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
