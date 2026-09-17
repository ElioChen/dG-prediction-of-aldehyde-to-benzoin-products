#!/bin/bash
#SBATCH --job-name=homo_sa_gnn_full
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/homo_sa_gnn_full_%j.out
#
# Full-library homo standalone GNN leg (CAMPAIGN_PLAN.md Phase 3): single
# attentive-pooling GNN on the ~184k homo table, same B97-3c-baseline Delta
# target as the tabular leg. CPU-trained (same as the r1-10 champion GNN).
# Submit with --dependency=afterok:<tabular jobid> once
# submit_homo_standalone_full.sh has written the table.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export CB_BASELINE_COL=dG_b973c_kcal

OUT=data/cross_benzoin/homo_standalone
TABLE=$OUT/homo_standalone_full_library_table_slim260.parquet
SEED=${SEED:-0}
echo "homo_sa_gnn_full seed=$SEED node=${SLURMD_NODENAME} $(date)"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table "$TABLE" \
    --champion-dir "$OUT/tabular_full" \
    --ensemble-path "$OUT/tabular_full/models/ensemble_scaffold_disjoint.joblib" \
    --outdir "$OUT/gnn_attentive_full_seed${SEED}" \
    --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed $SEED
RC=$?
echo "Done seed=$SEED $(date) exit=$RC"
exit $RC
