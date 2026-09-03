#!/bin/bash
#SBATCH --job-name=gnn_attn_7r_cpu
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/gnn_attn_7r_cpu_%j.out
#
# CPU fallback for the r1-7 recovered blend-GNN retrain. The gpu_h100/a100 queue
# is stuck on fairshare (account ugs25056 EffectvUsage 0.88, FairShare 0.02) --
# every freed GPU is backfill-planned for higher-priority users, so 26339586 sat
# PENDING(Priority) indefinitely. This model is small (attentive GINEConv, ~830k
# params, ~20k r1-7 train pairs, batch 64, <=150 epochs / patience 25) and NOT on
# the critical path, so a ~1-3h CPU run on wide-open fat_rome beats waiting.
#
set -o pipefail
REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/nequip/bin/python}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32

cd "$REPO"
echo "gnn_attentive_7rounds_recovered CPU node=${SLURMD_NODENAME} py=$PY $(date)"
$PY -c "import torch,torch_geometric;torch.set_num_threads(32);print('torch',torch.__version__,'tg',torch_geometric.__version__,'threads',torch.get_num_threads(),'cuda',torch.cuda.is_available())"

$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table data/cross_benzoin/cross_round7/cross_train_table_7rounds_recovered_scaffold_split_labeled_slim260.parquet \
    --champion-dir data/cross_benzoin/cross_round7/train_ensemble_7rounds_recovered_v1 \
    --ensemble-path data/cross_benzoin/cross_round7/train_ensemble_7rounds_recovered_v1/models/cross_ensemble_model.joblib \
    --outdir data/cross_benzoin/cross_round7/gnn_attentive_7rounds_recovered_v1 \
    --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed 0
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
