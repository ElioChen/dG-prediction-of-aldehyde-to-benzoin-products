#!/bin/bash
#SBATCH --job-name=promote_gnn_stack
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/promote_gnn_stack_%j.out
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
time cat /gpfs/scratch1/shared/schen3/envs/gnn_lite/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1
echo "promote GNN+tabular stacking to full library, node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
$PY -u "$REPO/pipeline/analysis/promote_gnn_stacking_full_library.py"
rc=$?
echo "Done $(date) exit=$rc"
exit $rc
