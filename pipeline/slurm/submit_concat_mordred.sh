#!/bin/bash
#SBATCH --job-name=concat_mordred
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/logs/concat_mordred_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "concat mordred chunks $(date)"
$PY -u cross_benzoin/analysis/concat_mordred_chunks.py
echo "Done $(date) exit=$?"
