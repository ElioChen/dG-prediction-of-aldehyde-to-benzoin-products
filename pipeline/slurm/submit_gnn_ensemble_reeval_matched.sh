#!/bin/bash
#SBATCH --job-name=gnn_reeval_matched
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/gnn_reeval_matched_%j.out
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "GNN re-eval on tabular-matched ids node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
$PY -u "$REPO/pipeline/analysis/gnn_ensemble_reeval_matched.py"
rc=$?
echo "Done $(date) exit=$rc"
exit $rc
