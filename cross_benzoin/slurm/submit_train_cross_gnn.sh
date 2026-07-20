#!/bin/bash
#SBATCH --job-name=cross_gnn
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#
# SLURM wrapper for cross_benzoin/train_cross_gnn.py -- the cross-benzoin GNN+
# tabular stacking prototype, mirroring homo's confirmed win (see
# pipeline/slurm/submit_gnn_dual_qm_champion275_aligned_v2.sh's conventions:
# envs/gnn_lite python, prewarm the torch .so files to dodge the known cold-GPFS-
# read slowness). Uses gpu_h100 (memory cluster_gpu_partitions: check h100 before
# defaulting to a100) -- also genuinely needed here since this is real GPU work,
# unlike the CPU-only jobs elsewhere in this project that should NOT be routed here.
#
# Submit:
#   sbatch --output="/scratch-shared/schen3/benzoin-dg/<outdir>/gnn_%j.out" \
#     cross_benzoin/slurm/submit_train_cross_gnn.sh
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"
TABLE="${TABLE:-data/cross_benzoin/cross_round5/cross_train_table_5rounds_mordred_slim120.parquet}"
CHAMPION_DIR="${CHAMPION_DIR:-data/cross_benzoin/cross_round5/train_5rounds_mordred_slim120_v1}"
ENSEMBLE_DIR="${ENSEMBLE_DIR:-data/cross_benzoin/cross_round5/train_ensemble_slim120_v1}"
OUTDIR="${OUTDIR:-data/cross_benzoin/cross_round5/train_gnn_v1}"
INIT_FROM_HOMO="${INIT_FROM_HOMO:-}"
EXTRA_VAL_FRAC="${EXTRA_VAL_FRAC:-0.0}"
LR="${LR:-0.001}"
PATIENCE="${PATIENCE:-15}"
SEED="${SEED:-0}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
time cat /gpfs/scratch1/shared/schen3/envs/gnn_lite/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1

echo "cross_gnn node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} lr=$LR extra_val_frac=$EXTRA_VAL_FRAC patience=$PATIENCE $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
cd "$REPO"
mkdir -p "$OUTDIR"
EXTRA_ARGS=(--lr "$LR" --patience "$PATIENCE" --seed "$SEED")
[ -n "$INIT_FROM_HOMO" ] && EXTRA_ARGS+=(--init-from-homo "$INIT_FROM_HOMO")
[ "$EXTRA_VAL_FRAC" != "0.0" ] && EXTRA_ARGS+=(--extra-val-frac "$EXTRA_VAL_FRAC")
$PY -u cross_benzoin/train_cross_gnn.py \
    --table "$TABLE" --champion-dir "$CHAMPION_DIR" --ensemble-dir "$ENSEMBLE_DIR" \
    --outdir "$OUTDIR" "${EXTRA_ARGS[@]}"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
