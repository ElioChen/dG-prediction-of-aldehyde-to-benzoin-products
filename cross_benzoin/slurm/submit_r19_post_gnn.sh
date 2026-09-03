#!/bin/bash
#SBATCH --job-name=r19_postgnn
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/r19_postgnn_%j.out
#
# Runs after the r1-9 attentive GNN (submit_r19_retrain_chain.sh submits this with
# --dependency=afterany:<gnn_jobid>). Three steps, each guarded and non-fatal:
#   1. verify_and_bootstrap_9rounds.py  -> r1-9 headline MAE + P(blend>ensemble)
#   2. score_round_active_learning.py   -> re-score round10 with the r1-9 table/model
#   3. build round10 SP geom list (top-2000) + sbatch the product r2SCAN-3c SP array
# Step 3 only fires if the selection CSV looks sane; otherwise it logs a MANUAL note
# and exits 0 -- no wasted DFT, nothing broken. round10 label assembly + the round10
# AL *decision* stay with the user (HANDOFF §1.4).
#
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nequip/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
echo "r19_postgnn node=${SLURMD_NODENAME:-?} $(date)"

R9=data/cross_benzoin/cross_round9
SLIM="$R9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet"
FEATLIST="$R9/scaffold_disjoint_9rounds_v1/models/feature_list.json"
GNNDIR="$R9/gnn_attentive_9rounds_v1"

# ---- 1. bootstrap verify -------------------------------------------------------
if [[ -f "$GNNDIR/models/metadata.json" || -f "$GNNDIR/models/gnn_state.pt" || -f "$GNNDIR/models/cross_gnn_state.pt" ]]; then
  echo "=== [1/3] verify_and_bootstrap_9rounds ==="
  "$PY" -u cross_benzoin/verify_and_bootstrap_9rounds.py || echo "  bootstrap verify exit $?"
else
  echo "=== [1/3] SKIP bootstrap -- no GNN artefacts in $GNNDIR ==="
fi

# ---- 2. re-score round10 with the r1-9 table --------------------------------------
SEL="data/cross_benzoin/cross_round10_fat20_stage1/cross_round10_fat20_stage1_dft_selection.csv"
if [[ -s "$SLIM" && -s "$FEATLIST" ]]; then
  echo "=== [2/3] score_round_active_learning (r1-9 table) ==="
  "$PY" -u cross_benzoin/score_round_active_learning.py \
      --tag cross_round10_fat20_stage1 \
      --candidates-path data/cross_benzoin/cross_round10_fat20_stage1/cross_round10_fat20_stage1_features_cross_round10_products_merged.parquet \
      --train-table "$SLIM" --feature-list "$FEATLIST" \
      --model ensemble --n-boot 40 --n-select 2000 --seed 42 || echo "  rescore exit $?"
else
  echo "=== [2/3] SKIP rescore -- r1-9 slim table / feature-list missing ==="
fi

# ---- 3. round10 product SP for the r1-9-selected top-2000 -----------------------
echo "=== [3/3] round10 SP submit ==="
if [[ -s "$SEL" ]] && head -1 "$SEL" | grep -q "dG_pred_correction_std"; then
  "$PY" cross_benzoin/build_round10_sp_geomlist.py --limit 2000
  LIST="$REPO/data/cross_benzoin/cross_round10_fat20_stage1/round10_sp_geom_list.csv"
  N=$(( $(wc -l < "$LIST") - 1 ))
  if [[ "$N" -ge 100 ]]; then
    OUT="$REPO/data/raw/dft_sp_cross/cross_round10_fat20_stage1/sp_products"
    CHUNK=100; NT=$(( (N + CHUNK - 1) / CHUNK )); mkdir -p "$OUT/logs"
    echo "  submitting round10 SP: $N geoms, $NT tasks %110"
    sbatch --array=0-$((NT-1))%110 --output="$OUT/logs/%A_%a.out" \
      --export=ALL,REPO="$REPO",GEOM_LIST="$LIST",OUTDIR="$OUT",CHUNK=$CHUNK \
      cross_benzoin/slurm/submit_r89_sp_array_fatrome.sh
  else
    echo "  MANUAL: geom list only $N rows -- round10 SP NOT submitted. Check $LIST"
  fi
else
  echo "  MANUAL: no sane $SEL -- round10 SP NOT submitted. Rescore by hand, then:"
  echo "    $PY cross_benzoin/build_round10_sp_geomlist.py --limit 2000 && sbatch --array=0-19%110 ... submit_r89_sp_array_fatrome.sh"
fi
echo "Done $(date)"
