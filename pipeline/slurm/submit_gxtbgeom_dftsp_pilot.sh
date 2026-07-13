#!/bin/bash
#SBATCH --job-name=gxtb_dft
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/gxtb_test/logs/gxtbdft_%j.out
#
# PILOT: g-xTB geometry -> (g-xTB Hessian thermo + r2SCAN-3c SP) for ΔG_gxtb and
# ΔG_dft//gxtb, vs the existing 1% ΔG_dft//xtb. 30 mols spanning classes. Serial
# ORCA (nproc 1, no MPI) parallelised across molecules (--workers 24), the proven
# thermo_orca pattern. Scale to full 1% only if the pilot shows a meaningful effect.
#
PROJ="/scratch-shared/schen3"; WD="$PROJ/gxtb_test"
PY="/home/schen3/venv/nhc-workflow/bin/python"
source /etc/profile 2>/dev/null
module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$WD/logs" "$WD/pilot_xyz_20260620"

echo "gxtbgeom->DFTsp PILOT  node=${SLURMD_NODENAME}  $(date)"
cd "$WD"
$PY -u gxtbgeom_dftsp.py \
    --input  "$WD/pilot_1pct_gxtbgeom_20260620.csv" \
    --out    "$WD/gxtbgeom_dftsp_pilot_20260620.csv" \
    --xyzdir "$WD/pilot_xyz_20260620" \
    --scratch "${TMPDIR:-$PROJ/tmp}" \
    --nproc 1 --workers 24
echo "Done $(date)"
