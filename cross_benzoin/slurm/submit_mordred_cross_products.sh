#!/bin/bash
#SBATCH --job-name=mordred_cross
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#
# Targeted-Mordred descriptors (MoRSE/CPSA/Polarizability/GeometricalIndex/
# MomentOfInertia/PBF/McGowanVolume/VdwVolumeABC/Weight/TopoPSA) for cross
# products, reusing already-saved geometry (no new xTB/DFT). ~1s/molecule.
#
# Submit:
#   IN=/abs/path/products.csv
#   OUT=/abs/path/mordred_products
#   N=$(wc -l < "$IN"); CHUNK=100; NCH=$(( (N+CHUNK-1)/CHUNK ))
#   mkdir -p "$OUT/logs"
#   sbatch --array=0-$((NCH-1))%48 --output="$OUT/logs/mrd_%a.out" \
#     --export=ALL,PRODUCTS="$IN",OUTDIR="$OUT",CHUNK=$CHUNK submit_mordred_cross_products.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
# nhc-workflow's mordred install is broken (empty namespace package, no __init__.py --
# `from mordred import Calculator` fails there). envs/bde_gnn has a working mordred +
# rdkit + pandas, confirmed 2026-07-15.
PY="/gpfs/scratch1/shared/schen3/envs/bde_gnn/bin/python"
PRODUCTS="${PRODUCTS:?set PRODUCTS=/abs/products.csv}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"
CHUNK="${CHUNK:-100}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
ID=${SLURM_ARRAY_TASK_ID:-0}
echo "mordred_cross chunk=$ID node=${SLURMD_NODENAME} $(date)"
cd "$REPO"
$PY -u cross_benzoin/add_mordred_cross_products.py \
    --products-csv "$PRODUCTS" --chunk-id "$ID" --chunk-size "$CHUNK" --out-dir "$OUTDIR"
echo "Done chunk=$ID $(date) exit=$?"
