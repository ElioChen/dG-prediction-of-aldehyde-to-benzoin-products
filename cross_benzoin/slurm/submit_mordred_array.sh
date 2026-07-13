#!/bin/bash
#SBATCH --job-name=mordred_desc
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --time=01:00:00
#
# Mordred descriptor set (1826, incl 3D WHIM/GETAWAY/MoRSE/RDF) for a CHUNK of homo_v6
# products_all.csv or aldehydes_all.csv, reusing saved xyz geometry. Serial within a task
# (ProcessPoolExecutor DEADLOCKS -- fork()+threaded-BLAS hang, see add_mordred_descriptors.py
# comment) -- speed comes from SLURM array-level parallelism (%128), not intra-task workers.
#
# Submit (WHICH=products|aldehydes):
#   WHICH=products; CHUNKSIZE=250
#   N=$(python -c "import pandas as pd; print(pd.read_csv('data/cross_benzoin/homo_v6/${WHICH}_all.csv', usecols=['id']).id.nunique())")
#   NCH=$(( (N+CHUNKSIZE-1)/CHUNKSIZE ))
#   OUT=data/cross_benzoin/homo_v6/mordred_${WHICH}; mkdir -p "$OUT/logs"
#   sbatch --array=0-$((NCH-1))%128 --output="$OUT/logs/mrd_%a.out" \
#     --export=ALL,WHICH=$WHICH,CHUNKSIZE=$CHUNKSIZE,OUTDIR=$OUT cross_benzoin/slurm/submit_mordred_array.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
WHICH="${WHICH:?set WHICH=products|aldehydes}"
CHUNKSIZE="${CHUNKSIZE:-250}"
OUTDIR="${OUTDIR:?set OUTDIR=/abs/out/dir}"

source /etc/profile 2>/dev/null
module load 2023 2>/dev/null

ID=$SLURM_ARRAY_TASK_ID
echo "mordred ${SLURM_ARRAY_JOB_ID}[$ID] which=$WHICH node=${SLURMD_NODENAME} $(date)"
cd "$REPO"
$PY -u cross_benzoin/analysis/add_mordred_descriptors.py \
    --which "$WHICH" --chunk-id "$ID" --chunk-size "$CHUNKSIZE" --out-dir "$OUTDIR"
echo "Done task=$ID $(date) exit=$?"
