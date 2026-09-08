#!/bin/bash
#SBATCH --job-name=gnn_seed_r10g
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=1-4
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/gnn_seed_r10c_%A_%a.out
#
# CPU version of submit_gnn_seed_ensemble_r10.sh -- GPU partitions (gpu_a100 /
# gpu_h100 / fat_rome) were all saturated 2026-09-08, GPU array 26464493 sat
# PENDING for hours. The r1-10 champion GNN itself was trained on CPU
# (submit_gnn_attentive_10rounds_cpu.sh), so this is a proven path; ~6-10 h/seed
# on 16 rome cores vs ~1.5 h on a GPU. Does not meaningfully compete with the Tier
# B campaign (rome had 120 idle nodes; this is 64 cores total).
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
R10=data/cross_benzoin/cross_round10
SEED=${SLURM_ARRAY_TASK_ID:-1}
OUT=$R10/gnn_attentive_10rounds_seed${SEED}
if [[ -f "$OUT/models/gnn_state.pt" ]]; then echo "seed $SEED already done, skip"; exit 0; fi
echo "gnn_seed_r10c seed=$SEED node=${SLURMD_NODENAME} $(date)"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table $R10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet \
    --champion-dir $R10/scaffold_disjoint_10rounds_v1 \
    --ensemble-path $R10/scaffold_disjoint_10rounds_v1/models/ensemble_scaffold_disjoint.joblib \
    --outdir $OUT \
    --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed $SEED
RC=$?
echo "Done seed=$SEED $(date) exit=$RC"
exit $RC
