#!/bin/bash
#SBATCH --job-name=verify_bde_champ
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/bde_gnn/bin/python"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"

ID=${SLURM_ARRAY_TASK_ID:-0}
WHICH=aldehydes
[ "$ID" -eq 1 ] && WHICH=products

echo "verify_bde_champion which=$WHICH node=${SLURMD_NODENAME} $(date)"
$PY -u pipeline/bde/predict_bde_champion.py --which "$WHICH" --verify
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
