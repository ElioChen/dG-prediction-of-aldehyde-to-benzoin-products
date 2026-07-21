#!/bin/bash
#SBATCH --job-name=unif_r7_check
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Re-test homo+cross unification at round1-7 scale (old, non-scaffold CV split),
# mirroring the round1-5/round1-6 unification checks. See
# cross_benzoin/analysis/unification_check_round7.py for full method docstring.
#
REPO="/gpfs/scratch1/shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "unification_check_round7 node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/analysis/unification_check_round7.py \
    --unified-table data/cross_benzoin/homo_unify/cross_train_table_unified_v2_7rounds_mordred.parquet \
    --feature-list data/cross_benzoin/cross_round7/train_7rounds_mordred_slim120_v1/models/feature_list.json \
    --cross-only-cv-mae 2.122136494840926 \
    --outdir data/cross_benzoin/cross_round7/unification_check_v1
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
