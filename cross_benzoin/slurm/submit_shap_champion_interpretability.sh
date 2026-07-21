#!/bin/bash
#SBATCH --job-name=shap_champion
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "shap_champion_interpretability node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/analysis/shap_champion_interpretability.py \
    --model-dir data/cross_benzoin/cross_round7/scaffold_disjoint_v1 \
    --table data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_split_labeled.parquet \
    --outdir data/cross_benzoin/cross_round7/shap_champion_v1
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
