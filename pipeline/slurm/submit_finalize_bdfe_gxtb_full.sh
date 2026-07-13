#!/bin/bash
#SBATCH --job-name=finalize_bdfe_gxtb_full
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/finalize_bdfe_gxtb_full_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "FINALIZE g-xTB-consistent BDFE/BDE full (aldehyde+product) comparison $(date)"
$PY -u "$REPO/pipeline/analysis/finalize_correction_bdfe_gxtb_full.py"
echo "Done $(date) exit=$?"
