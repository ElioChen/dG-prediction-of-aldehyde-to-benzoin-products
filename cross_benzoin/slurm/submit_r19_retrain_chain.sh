#!/bin/bash
#SBATCH --job-name=r19_retrain
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/r19_retrain_%j.out
#
# Steps 1-4 of the r1-9 retrain (assemble mordred table -> relabel scaffold split ->
# prune to frozen 260-feat schema -> train_scaffold_disjoint champion+ensemble), then
# submits the r1-9 attentive GNN retrain. Auto-chained after the assemble job
# (submit_r89_assemble_after_ald_sp.sh) so the recovery reaches a retrained r1-9
# model with no live session. Non-fatal: a step failing is logged and the job exits
# non-zero, leaving run_r19_retrain_chain.sh to be finished by hand.
#
#   sbatch --dependency=afterok:<assemble_jobid> cross_benzoin/slurm/submit_r19_retrain_chain.sh
#
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
cd "$REPO"
export PY=/home/schen3/venv/nequip/bin/python   # has sklearn/xgboost/pyarrow/rdkit/torch_geometric
echo "r19_retrain node=${SLURMD_NODENAME:-?} $(date)"

for r in 8 9; do
  f="data/raw/dft_sp_cross/cross_round${r}/cross_round${r}_dft_sp.csv"
  [[ -s "$f" ]] || { echo "ABORT: $f missing/empty -- assemble step did not produce labels"; exit 1; }
done

bash cross_benzoin/run_r19_retrain_chain.sh
RC=$?
echo "run_r19_retrain_chain.sh exit=$RC $(date)"
[[ $RC -eq 0 ]] || { echo "chain steps 1-4 failed; NOT submitting GNN"; exit $RC; }

R9=data/cross_benzoin/cross_round9
SLIM="$R9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet"
CH="$R9/scaffold_disjoint_9rounds_v1"
if [[ -s "$SLIM" && -d "$CH" ]]; then
  echo "submitting r1-9 attentive GNN retrain"
  GNN_JID=$(sbatch --parsable --job-name=gnn_attn_9r_cpu --partition=fat_rome --nodes=1 --ntasks=1 \
    --cpus-per-task=32 --mem=120G --time=10:00:00 \
    --output="$REPO/slurm_logs/gnn_attn_9r_cpu_%j.out" \
    --wrap="source /etc/profile; module load 2023; export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32; cd $REPO && $PY -u cross_benzoin/train_cross_gnn_arch_sweep.py --table $SLIM --champion-dir $CH --ensemble-path $CH/models/ensemble_scaffold_disjoint.joblib --outdir $R9/gnn_attentive_9rounds_v1 --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed 0")
  echo "  GNN job = $GNN_JID"
  if [[ -n "$GNN_JID" ]]; then
    POST_JID=$(sbatch --parsable --dependency=afterany:"$GNN_JID" \
      cross_benzoin/slurm/submit_r19_post_gnn.sh)
    echo "  post-GNN (bootstrap + round10 rescore + round10 SP) job = $POST_JID (afterany:$GNN_JID)"
  fi
else
  echo "SLIM/champion-dir missing after chain -- GNN not submitted"
fi
echo "Done $(date)"
