#!/bin/bash
#SBATCH --job-name=bdfe2_desc
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#
# BDFE v2 (optimized): reuses the parent's already-computed G_xtb/xtb_energy (from
# {aldehydes,products}_all.csv, DMSO/ALPB solvent, same as the rest of this project) --
# only the two homolysis radical fragments get a fresh --ohess --alpb dmso. Solvent
# consistency validated (job 24416958) after catching+fixing a parent-DMSO/fragment-gas-
# phase mismatch in the first cut. Cheaper than v1 (calc_bde_free_energy.py): v1 wastefully
# recomputed the parent from scratch every time.
#
# Submit (WHICH=aldehydes|products):
#   WHICH=aldehydes; CHUNKSIZE=150
#   OUT=data/cross_benzoin/homo_v6/bdfe2_${WHICH}; mkdir -p "$OUT/logs"
#   sbatch --array=0-NCH%128 --output="$OUT/logs/bdfe2_%a.out" \
#     --export=ALL,WHICH=$WHICH,CHUNKSIZE=$CHUNKSIZE,OUTDIR=$OUT pipeline/slurm/submit_bdfe_v2_array.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
XTB_BIN="/home/schen3/xtb/bin/xtb"
WHICH="${WHICH:?set WHICH=aldehydes|products}"
CHUNKSIZE="${CHUNKSIZE:-150}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"

ID=$SLURM_ARRAY_TASK_ID
echo "bdfe2 ${SLURM_ARRAY_JOB_ID}[$ID] which=$WHICH node=${SLURMD_NODENAME} $(date)"
cd "$REPO/pipeline/compute"
WORK="${TMPDIR:-$REPO/tmp}/bdfe2_${SLURM_ARRAY_JOB_ID}_${ID}"
$PY -u calc_bde_free_energy_v2.py --which "$WHICH" --chunk-id "$ID" --chunk-size "$CHUNKSIZE" \
    --out-dir "$OUTDIR" --xtb-bin "$XTB_BIN" --work-dir "$WORK"
rm -rf "$WORK"
echo "Done task=$ID $(date) exit=$?"
