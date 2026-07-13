#!/bin/bash
#SBATCH --job-name=finalize_hypertags275
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/finalize_hypertags275_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/bde_lite/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "FINALIZE champion275 + explicit hypervalent SMARTS tags (Action D, Tier1b) $(date)"
$PY -u "$REPO/pipeline/analysis/finalize_correction_hypervalent_tags.py"
rc=$?
echo "Done $(date) exit=$rc"
exit $rc
