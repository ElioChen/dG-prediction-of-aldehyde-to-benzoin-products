#!/bin/bash
#SBATCH --job-name=assemble_feat
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# SLURM wrapper for assemble_cross_round_features.py (RDKit descriptors + mordred join
# against the 220k-molecule aldehyde library -- too slow to run in the login-node
# foreground, found 2026-07-20 on round8).
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"
TAG="${TAG:?set TAG}"
MORDRED_GLOB="${MORDRED_GLOB:-}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "assemble_cross_round_features tag=$TAG node=${SLURMD_NODENAME} $(date)"
if [ -n "$MORDRED_GLOB" ]; then
  $PY -u cross_benzoin/assemble_cross_round_features.py --tag "$TAG" --product-mordred-csv $MORDRED_GLOB
else
  $PY -u cross_benzoin/assemble_cross_round_features.py --tag "$TAG"
fi
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
