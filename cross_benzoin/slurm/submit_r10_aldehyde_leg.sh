#!/bin/bash
#SBATCH --job-name=r10_ald_leg
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/r10_ald_leg_%j.out
#
# Runs after the round10 aldehyde funnel_v3 regen array (--dependency=afterany:<regen>).
# Builds the aldehyde SP geom list + thermal from the regen output, then submits the
# round10 aldehyde r2SCAN-3c SP array. Mirrors the r89 aldehyde leg.
#
#   sbatch --dependency=afterany:<regen_jobid> cross_benzoin/slurm/submit_r10_aldehyde_leg.sh
#
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
cd "$REPO"
echo "r10_ald_leg node=${SLURMD_NODENAME:-?} $(date)"

REGEN="$REPO/data/cross_benzoin/r10_aldehyde_regen"
RECOV="$REPO/data/cross_benzoin/r10_aldehyde_recover"
"$PY" cross_benzoin/build_r89_aldehyde_geomlist_regen.py \
  --regen-dir "$REGEN" --out-dir "$RECOV" --check-xyz
LIST="$RECOV/aldehyde_geom_list.csv"
N=$(( $(wc -l < "$LIST") - 1 ))
echo "geom list: $N aldehydes"
[[ "$N" -ge 50 ]] || { echo "ABORT: geom list too small ($N)"; exit 1; }

OUT="$REPO/data/raw/dft_sp_cross/r10_aldehyde_sp"
CHUNK=100; NT=$(( (N + CHUNK - 1) / CHUNK )); mkdir -p "$OUT/logs"
echo "submitting round10 aldehyde SP: $NT tasks %110"
sbatch --array=0-$((NT-1))%110 --output="$OUT/logs/%A_%a.out" \
  --export=ALL,REPO="$REPO",GEOM_LIST="$LIST",OUTDIR="$OUT",CHUNK=$CHUNK \
  cross_benzoin/slurm/submit_r89_sp_array_fatrome.sh
echo "Done $(date)"
