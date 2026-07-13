#!/bin/bash
#SBATCH --job-name=finalize_morfeus
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/finalize_morfeus_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "FINALIZE morfeus-augmented comparison $(date)"
$PY -u "$REPO/pipeline/analysis/finalize_correction_morfeus.py"
echo "Done $(date) exit=$?"
