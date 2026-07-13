#!/bin/bash
#SBATCH --job-name=concat_bde
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/logs/concat_bde_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "concat bde chunks (which=${WHICH:-both}) $(date)"
$PY -u pipeline/compute/concat_bde_chunks.py ${WHICH:+--which $WHICH}
echo "Done $(date) exit=$?"
