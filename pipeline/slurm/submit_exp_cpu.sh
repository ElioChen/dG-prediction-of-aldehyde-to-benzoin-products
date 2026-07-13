#!/bin/bash
#SBATCH --job-name=exp_cpu
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/exp_%x_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "EXP ${SCRIPT} node=${SLURMD_NODENAME} $(date)"
$PY -u "$REPO/pipeline/analysis/${SCRIPT}"
echo "Done $(date) exit=$?"
