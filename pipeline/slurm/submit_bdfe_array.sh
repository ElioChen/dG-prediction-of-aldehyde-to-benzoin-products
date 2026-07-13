#!/bin/bash
#SBATCH --job-name=bdfe_desc
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#
# BDFE (free-energy corrected BDE: xtb --ohess RRHO + optional Shermo qRRHO) for a CHUNK of
# homo_v6, reusing saved xyz geometry. Pilot (20 mols, 100% success) validated the method
# and the uhf=0(parent)/uhf=1(fragment) fix. Smaller chunk size than the raw-BDE array
# (100 vs 200) since --ohess is costlier than --opt; array-level parallelism (%128) for
# wall-clock speed, same pattern as calc_bde.py.
#
# Submit (WHICH=aldehydes|products):
#   WHICH=aldehydes; CHUNKSIZE=100
#   OUT=data/cross_benzoin/homo_v6/bdfe_${WHICH}; mkdir -p "$OUT/logs"
#   sbatch --array=0-NCH%128 --output="$OUT/logs/bdfe_%a.out" \
#     --export=ALL,WHICH=$WHICH,CHUNKSIZE=$CHUNKSIZE,OUTDIR=$OUT pipeline/slurm/submit_bdfe_array.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
XTB_BIN="/home/schen3/xtb/bin/xtb"
SHERMO_BIN="/home/schen3/.local/bin/Shermo"
WHICH="${WHICH:?set WHICH=aldehydes|products}"
CHUNKSIZE="${CHUNKSIZE:-100}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"

ID=$SLURM_ARRAY_TASK_ID
echo "bdfe ${SLURM_ARRAY_JOB_ID}[$ID] which=$WHICH node=${SLURMD_NODENAME} $(date)"
cd "$REPO/pipeline/compute"
WORK="${TMPDIR:-$REPO/tmp}/bdfe_${SLURM_ARRAY_JOB_ID}_${ID}"
$PY -u calc_bde_free_energy.py --which "$WHICH" --chunk-id "$ID" --chunk-size "$CHUNKSIZE" \
    --out-dir "$OUTDIR" --xtb-bin "$XTB_BIN" --shermo-bin "$SHERMO_BIN" --work-dir "$WORK"
rm -rf "$WORK"
echo "Done task=$ID $(date) exit=$?"
