#!/bin/bash
#SBATCH --job-name=deep_error_champion275
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/deep_error_champion275_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "Deep error / noise-band analysis champion275 $(date)"
$PY -u "$REPO/pipeline/analysis/deep_error_analysis_champion275.py"
rc=$?
echo "Done $(date) exit=$rc"
exit $rc
