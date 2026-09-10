#!/bin/bash
#SBATCH --job-name=homo_regen
#SBATCH --partition=fat_genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-770%80
#SBATCH --requeue
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/data/cross_benzoin/homo_standalone/relabel_sp/regen_logs/hr_%A_%a.out
#
# GAP-FILL for the homo SP relabel: pairs whose archived geometry is missing
# (chunks ~1400-1830 partial + ~59 chunks with no zst). Here we DO regenerate
# the geometry: conf_funnel_v3 + GFN2 --ohess (fresh thermal) + r2SCAN-3c +
# B97-3c + g-xTB SP per species -> homo dG = G_prod - 2*G_ald. Reuses
# rec_homo_relabel_worker.py (2-species homo variant).
#
# Prereq: cross_benzoin/split_homo_sp_manifest.py has produced
#   relabel_sp/homo_sp_manifest_regen.csv, then adapt it to the worker schema:
#   python -c "import pandas as pd; d=pd.read_csv('.../homo_sp_manifest_regen.csv');
#     d=d.rename(columns={'id':'pid','ald_smiles':'donor_smiles','label_stored':'label'});
#     d['acceptor_smiles']=d['donor_smiles']; d['grp']='homo'; d['gxtb_stored']=float('nan');
#     d.to_csv('.../homo_sp_regen_pairs.csv',index=False)"
# Set --array to ceil(N_regen / CHUNK).  CHUNK small: each pair ~2-3 CPU-h (geom+3SP x2 species).
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
SAMP="$REPO/data/cross_benzoin/homo_standalone/relabel_sp/homo_sp_regen_pairs.csv"
OUTD="$REPO/data/cross_benzoin/homo_standalone/relabel_sp/regen_shards"
CHUNK=6
mkdir -p "$OUTD" "$REPO/data/cross_benzoin/homo_standalone/relabel_sp/regen_logs"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export PATH="/home/schen3/xtb/bin:/home/schen3/orca:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XTB_BIN=/home/schen3/xtb/bin/xtb XTBPATH=/home/schen3/xtb/share/xtb
export ORCA_BIN=/home/schen3/orca/orca ORCA_SCF=default
cd "$REPO"
ID=${SLURM_ARRAY_TASK_ID:-0}
START=$(( ID * CHUNK ))
if [[ -d /scratch-local ]]; then SCR="/scratch-local/${USER}.${SLURM_JOB_ID}_${ID}"; else SCR="/tmp/${USER}.${SLURM_JOB_ID}_${ID}"; fi
mkdir -p "$SCR"
OUT="$OUTD/shard_$(printf '%05d' "$ID").csv"
if [[ -f "$OUT.done" ]]; then echo "task $ID done, skip"; exit 0; fi
$PY -u "$REPO/cross_benzoin/rec_homo_relabel_worker.py" \
    --sample "$SAMP" --skip "$START" --nrows "$CHUNK" --out "$OUT" --scratch "$SCR" --xtb-cores 7
RC=$?
rm -rf "$SCR" 2>/dev/null
if [[ $RC -eq 0 ]]; then
  NROWS=$(( $(wc -l < "$OUT") - 1 )); EXPECT=$CHUNK
  TOT=$(( $(wc -l < "$SAMP") - 1 ))
  [[ $(( START + CHUNK )) -gt $TOT ]] && EXPECT=$(( TOT - START ))
  [[ "$NROWS" -ge "$EXPECT" ]] && touch "$OUT.done"
fi
echo "task $ID rc=$RC rows=$(( $(wc -l < "$OUT") - 1 )) $(date)"
exit $RC
