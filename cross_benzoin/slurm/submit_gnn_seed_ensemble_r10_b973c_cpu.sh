#!/bin/bash
#SBATCH --job-name=gnn_seed_r10b973c
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=1-4
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/gnn_seed_r10b973c_%A_%a.out
#
# B97-3c-baseline variant of submit_gnn_seed_ensemble_r10_cpu.sh, per
# rec1_b973c_tierB/DRAIN_RUNBOOK.md step 4. Table/champion-dir repointed at the
# Tier B slim257 b973c retrain; CB_TARGET_COL/CB_BASELINE_COL set so the GNN
# trains on the same Delta-target as the new champion (dG_r2scan_kcal -
# dG_b973c_kcal). rome (not fat_rome): per CLUSTER_SHARING.md, fat partitions
# were saturated/expensive; rome had headroom 2026-09-10/11/14.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export CB_TARGET_COL=dG_r2scan_kcal CB_BASELINE_COL=dG_b973c_kcal
R10=data/cross_benzoin/cross_round10
NEW=$R10/scaffold_disjoint_10rounds_b973c_v1
SEED=${SLURM_ARRAY_TASK_ID:-1}
OUT=$R10/gnn_attentive_10rounds_b973c_seed${SEED}
if [[ -f "$OUT/models/gnn_state.pt" ]]; then echo "seed $SEED already done, skip"; exit 0; fi
echo "gnn_seed_r10b973c seed=$SEED node=${SLURMD_NODENAME} $(date)"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
$PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \
    --table $R10/cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet \
    --champion-dir $NEW \
    --ensemble-path $NEW/models/ensemble_scaffold_disjoint.joblib \
    --outdir $OUT \
    --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed $SEED
RC=$?
echo "Done seed=$SEED $(date) exit=$RC"
exit $RC
