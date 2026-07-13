#!/bin/bash
#SBATCH --job-name=shap_baseline72
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/shap_baseline72_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "SHAP baseline_72 deep interpretability $(date)"
$PY -u "$REPO/pipeline/analysis/shap_baseline72.py"
echo "Done $(date) exit=$?"
