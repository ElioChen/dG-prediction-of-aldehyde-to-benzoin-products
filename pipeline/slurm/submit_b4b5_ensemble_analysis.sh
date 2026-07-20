#!/bin/bash
#SBATCH --job-name=b4b5_ensemble
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/b4b5_ensemble_%j.out
#
# Runs pipeline/bde/analyze_b4b5_ensemble_residuals.py once the B4 (train_gnn_bde.py) and
# B5 (train_bondnet_bde.py) --pred-out reruns land. Submit with a dependency on those job
# IDs, e.g.:
#   sbatch --dependency=afterany:JID1:JID2:JID3 pipeline/slurm/submit_b4b5_ensemble_analysis.sh
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_lite"
mkdir -p "$REPO/runs/logs"

echo "B4+B5 ensemble + residual analysis $(date)"
$ENV/bin/python -u "$REPO/pipeline/bde/analyze_b4b5_ensemble_residuals.py" \
    --out "$REPO/runs/logs/b4b5_ensemble_residuals_result.json"
echo "done $(date) exit=$?"
