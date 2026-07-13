#!/bin/bash
#SBATCH --job-name=shap_subset_hard275
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/shap_subset_hard275_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/bde_lite/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "SHAP subset (sulfonyl/P/imine/amide hard tail) champion275 $(date)"
$PY -u "$REPO/pipeline/analysis/shap_subset_hypervalent_champion275.py"
rc=$?
echo "Done $(date) exit=$rc"
exit $rc
