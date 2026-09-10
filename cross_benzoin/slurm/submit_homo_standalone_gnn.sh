#!/bin/bash
#SBATCH --job-name=homo_sa_gnn
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/homo_sa_gnn_%j.out
#
# Standalone homo-only dG model, GNN leg: attentive-pooling TripleGNN on the
# same 30k homo table + scaffold-disjoint split, reusing the tabular job's
# champion-dir / ensemble for the blend stacking check. CPU (the r1-10 champion
# GNN was also CPU-trained). Submit with --dependency=afterok:<tabular jobid>.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

OUT=data/cross_benzoin/homo_standalone
TABLE=$OUT/homo_standalone_train_table_slim260.parquet
SEED=${SEED:-0}
echo "homo_sa_gnn seed=$SEED node=${SLURMD_NODENAME} $(date)"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table "$TABLE" \
    --champion-dir "$OUT/tabular" \
    --ensemble-path "$OUT/tabular/models/ensemble_scaffold_disjoint.joblib" \
    --outdir "$OUT/gnn_attentive_seed${SEED}" \
    --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed $SEED
RC=$?
echo "Done seed=$SEED $(date) exit=$RC"
exit $RC
