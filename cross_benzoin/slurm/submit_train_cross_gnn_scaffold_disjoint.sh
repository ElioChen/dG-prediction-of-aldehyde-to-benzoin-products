#!/bin/bash
#SBATCH --job-name=gnn_scafdis
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#
# SLURM wrapper for cross_benzoin/train_cross_gnn_scaffold_disjoint.py.
#
REPO="/scratch-shared/schen3/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"
TABLE="${TABLE:?set TABLE}"
CHAMPION_DIR="${CHAMPION_DIR:?set CHAMPION_DIR}"
ENSEMBLE_PATH="${ENSEMBLE_PATH:?set ENSEMBLE_PATH}"
OUTDIR="${OUTDIR:?set OUTDIR}"
INIT_FROM_HOMO="${INIT_FROM_HOMO:-}"
LR="${LR:-0.0003}"
PATIENCE="${PATIENCE:-25}"
EXTRA_VAL_FRAC="${EXTRA_VAL_FRAC:-0.1}"
SEED="${SEED:-0}"

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
time cat /gpfs/scratch1/shared/schen3/envs/gnn_lite/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1

echo "gnn_scaffold_disjoint node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} lr=$LR $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
cd "$REPO"
mkdir -p "$OUTDIR"
EXTRA_ARGS=(--lr "$LR" --patience "$PATIENCE" --extra-val-frac "$EXTRA_VAL_FRAC" --seed "$SEED")
[ -n "$INIT_FROM_HOMO" ] && EXTRA_ARGS+=(--init-from-homo "$INIT_FROM_HOMO")
$PY -u cross_benzoin/train_cross_gnn_scaffold_disjoint.py \
    --table "$TABLE" --champion-dir "$CHAMPION_DIR" --ensemble-path "$ENSEMBLE_PATH" \
    --outdir "$OUTDIR" "${EXTRA_ARGS[@]}"
RC=$?
echo "Done $(date) exit=$RC"
exit "$RC"
