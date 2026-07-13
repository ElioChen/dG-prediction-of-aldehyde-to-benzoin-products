#!/bin/bash
#SBATCH --job-name=bde_desc
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#
# BDE (aldehyde C-H or product new-C-C) for a CHUNK of homo_v6, reusing saved xyz geometry.
# Pilot (40 mols, 100% success, ~0.8s/mol) validated the method; array-level parallelism
# (%128) for wall-clock speed, same pattern as the mordred/RDKit backfills.
#
# Submit (WHICH=aldehydes|products):
#   WHICH=aldehydes; CHUNKSIZE=200
#   OUT=data/cross_benzoin/homo_v6/bde_${WHICH}; mkdir -p "$OUT/logs"
#   sbatch --array=0-NCH%128 --output="$OUT/logs/bde_%a.out" \
#     --export=ALL,WHICH=$WHICH,CHUNKSIZE=$CHUNKSIZE,OUTDIR=$OUT cross_benzoin/slurm/submit_bde_array.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
XTB_BIN="/home/schen3/xtb/bin/xtb"
WHICH="${WHICH:?set WHICH=aldehydes|products}"
CHUNKSIZE="${CHUNKSIZE:-200}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"

ID=$SLURM_ARRAY_TASK_ID
echo "bde ${SLURM_ARRAY_JOB_ID}[$ID] which=$WHICH node=${SLURMD_NODENAME} $(date)"
cd "$REPO/pipeline/compute"
WORK="${TMPDIR:-$REPO/tmp}/bde_${SLURM_ARRAY_JOB_ID}_${ID}"
$PY -u calc_bde.py --which "$WHICH" --chunk-id "$ID" --chunk-size "$CHUNKSIZE" \
    --out-dir "$OUTDIR" --xtb-bin "$XTB_BIN" --work-dir "$WORK"
rm -rf "$WORK"
echo "Done task=$ID $(date) exit=$?"
