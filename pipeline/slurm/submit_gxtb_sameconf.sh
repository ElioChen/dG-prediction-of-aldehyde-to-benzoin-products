#!/bin/bash
#SBATCH --job-name=gxtb_sameconf
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/schen3/gxtb_test/logs/sameconf_%j.out
PROJ="/scratch-shared/schen3"; WD="$PROJ/gxtb_test"
PY="/home/schen3/venv/nhc-workflow/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p "$WD/logs"
echo "same-conformer product geom test $(date)"
cd "$WD"
$PY -u gxtb_sameconf_product.py \
  --input "$WD/pilot_1pct_gxtbgeom_20260620.csv" \
  --xyzsrc "$PROJ/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c/benzoin_xyz" \
  --out "$WD/gxtb_sameconf_pilot_20260620.csv" --workers 24
echo "Done $(date)"
