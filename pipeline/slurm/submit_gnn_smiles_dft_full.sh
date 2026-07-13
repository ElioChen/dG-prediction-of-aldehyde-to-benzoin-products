#!/bin/bash
#SBATCH --job-name=gnn_smiles_dft_full
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/gnn_smiles_dft_full_%j.out
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/gpfs/scratch1/shared/schen3/envs/gnn_lite/bin/python"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
time cat /gpfs/scratch1/shared/schen3/envs/gnn_lite/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1
echo "pure-SMILES GINE, full-library real DFT labels, node=${SLURMD_NODENAME} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY -u "$REPO/pipeline/analysis/gnn_smiles_dft_full.py"
rc=$?
echo "Done $(date) exit=$rc"
exit $rc
