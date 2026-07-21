#!/bin/bash
#SBATCH --job-name=train_scafdis721
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Retrain champion (single-XGB) + MLP+XGB ensemble on the 70/20/10-molecule-level
# scaffold-disjoint split (Option A, user-approved 2026-07-20 -- see
# cross_benzoin/rebuild_scaffold_disjoint_split_v2.py for why this differs from the
# original 80/10/10 rebuild). Mirrors submit_train_scaffold_disjoint.sh exactly, just
# pointed at the new _721 table/outdir.
#
REPO="/gpfs/scratch1/shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "train_scaffold_disjoint_721 node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/train_scaffold_disjoint.py \
    --table data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_split_labeled_721.parquet \
    --outdir data/cross_benzoin/cross_round7/scaffold_disjoint_721_v1
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
