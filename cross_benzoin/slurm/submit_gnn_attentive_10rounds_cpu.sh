#!/bin/bash
#SBATCH --job-name=gnn_attn_10r_cpu
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/gnn_attn_10r_cpu_%j.out
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
R10=data/cross_benzoin/cross_round10
echo "gnn_attn_10r_cpu node=${SLURMD_NODENAME} $(date)"
$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table $R10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet \
    --champion-dir $R10/scaffold_disjoint_10rounds_v1 \
    --ensemble-path $R10/scaffold_disjoint_10rounds_v1/models/ensemble_scaffold_disjoint.joblib \
    --outdir $R10/gnn_attentive_10rounds_v1 \
    --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed 0
RC=$?
echo "GNN done $(date) exit=$RC"
[[ $RC -eq 0 ]] || exit $RC
echo "=== bootstrap ==="
$PY -u cross_benzoin/verify_and_bootstrap_10rounds.py
echo "Done $(date)"
