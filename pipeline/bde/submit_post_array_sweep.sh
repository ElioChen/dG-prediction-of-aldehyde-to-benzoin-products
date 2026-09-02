#!/bin/bash
#SBATCH --job-name=bde_post_sweep
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/bde_post_sweep_%j.out
#
# Unattended driver: when the two BDE featurize arrays finish, assemble the descriptor
# libraries and launch the scaffold-disjoint model sweep. Submitted with
#   sbatch --dependency=afterany:<genoa_arrayjob>:<rome_arrayjob> \
#          pipeline/bde/submit_post_array_sweep.sh
# so it sits in the queue until both arrays terminate (any state), then runs once.
#
# Safety gate: if fewer than MIN_FRAC of the 2209 chunks have a complete features.csv,
# it does NOT assemble (a crashed/partial array must not silently produce a short
# library) -- it logs loudly and exits 3 for a human to look at.
#
# Full plan / rationale: pipeline/bde/POST_ARRAY_MODEL_SWEEP.md
set -o pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
PY="${PY:-/home/schen3/venv/nhc-workflow/bin/python}"
CHUNK_DIR="${CHUNK_DIR:-$REPO/data/cross_benzoin/bde_homo_product_featurize_20260902}"
H="$REPO/data/cross_benzoin/homo_v6"
BK="${BK:-/home/schen3/benzoin_backups/bde_homo_rebuild_20260902}"
NCHUNK="${NCHUNK:-2209}"
MIN_FRAC="${MIN_FRAC:-0.97}"
SWEEP="${SWEEP:-1}"                 # 0 = assemble + backup only, don't submit GPU jobs

source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
cd "$REPO"
echo "=== bde_post_sweep start $(date) host=$(hostname) ==="

# ---- 1. completeness gate -------------------------------------------------------
complete=0
for c in "$CHUNK_DIR"/chunk_*; do
  [ -f "$c/features.csv" ] && [ -f "$c/pairs.csv" ] || continue
  p=$(( $(wc -l < "$c/pairs.csv") - 1 )); f=$(( $(wc -l < "$c/features.csv") - 1 ))
  [ "$p" -ge 1 ] && [ "$f" -ge "$p" ] && complete=$((complete+1))
done
need=$(awk -v n="$NCHUNK" -v fr="$MIN_FRAC" 'BEGIN{printf "%d", n*fr}')
echo "complete chunks: $complete / $NCHUNK  (need >= $need)"
if [ "$complete" -lt "$need" ]; then
  echo "GATE FAILED: too few complete chunks -- NOT assembling. Investigate the arrays."
  echo "  (resubmit this script by hand once the featurize arrays are actually done.)"
  exit 3
fi

# ---- 2. assemble --------------------------------------------------------------
echo "=== assemble_homo_descriptor_libs.py $(date) ==="
$PY pipeline/bde/assemble_homo_descriptor_libs.py --chunk-dir "$CHUNK_DIR" || { echo "assemble FAILED rc=$?"; exit 4; }
for f in products_all.csv aldehydes_all.csv; do
  [ -s "$H/$f" ] || { echo "MISSING $H/$f after assemble"; exit 4; }
  echo "  $H/$f : $(( $(wc -l < "$H/$f") - 1 )) rows"
done

# ---- 3. backup ---------------------------------------------------------------
mkdir -p "$BK"
tar -C "$H" -czf "$BK/products_all_$(date +%Y%m%d).csv.tar.gz" products_all.csv
tar -C "$H" -czf "$BK/aldehydes_all_full_$(date +%Y%m%d).csv.tar.gz" aldehydes_all.csv
sha256sum "$H"/products_all.csv "$H"/aldehydes_all.csv >> "$BK/SHA256SUMS.txt"
echo "backed up new libs -> $BK"

if [ "$SWEEP" != 1 ]; then echo "SWEEP=0, stopping after assemble+backup."; exit 0; fi

# ---- 4. launch the scaffold-disjoint model sweep --------------------------------
echo "=== submitting model sweep $(date) ==="
cd "$REPO"
j1=$(sbatch --parsable pipeline/slurm/submit_b6_scaffold_disjoint_ckpt.sh) && echo "  b6 ckpt        -> $j1"
j2=$(sbatch --parsable pipeline/slurm/submit_b6_deep_ensemble.sh)          && echo "  b6 deep ens    -> $j2"
j3=$(sbatch --parsable pipeline/slurm/submit_b4_b5_scaffold_disjoint.sh)   && echo "  b4/b5          -> $j3"
# aggregate the ensemble after it finishes
sbatch --parsable --dependency=afterany:$j2 \
  --job-name=b6_ens_agg --partition=staging --time=01:00:00 --mem=16G \
  --output="$REPO/slurm_logs/b6_ens_agg_%j.out" \
  --wrap "$PY $REPO/pipeline/bde/aggregate_b6_ensemble.py --ens-dir $REPO/runs/logs/scaffold_disjoint_bde/ensemble" \
  && echo "  ensemble agg queued (after $j2)"
# full-scale GBM bake-off (CPU) -- split paths expanded here, at submit time
ALD_SPLIT="$H/aldehydes_scaffold_split_from_dG.csv"
PROD_SPLIT="$H/products_scaffold_split.csv"
OUTD="$REPO/runs/logs/scaffold_disjoint_bde"
sbatch --parsable --job-name=gbm_bakeoff --partition=rome --time=04:00:00 --mem=32G --cpus-per-task=16 \
  --output="$REPO/slurm_logs/gbm_bakeoff_%j.out" \
  --wrap "cd $REPO && $PY pipeline/bde/gbm_bakeoff_hspoc.py --which aldehydes --split-file $ALD_SPLIT --out $OUTD/gbm_bakeoff_aldehydes_full.json && $PY pipeline/bde/gbm_bakeoff_hspoc.py --which products --split-file $PROD_SPLIT --out $OUTD/gbm_bakeoff_products_full.json" \
  && echo "  gbm bakeoff (full) queued"

echo "=== bde_post_sweep done $(date). Sweep jobs submitted; results land in runs/logs/scaffold_disjoint_bde/ ==="
echo "NEXT (human): git add -f the resulting *.json / *_pred.csv / *.pt, update pipeline/bde/STATUS.md ranking, rerun build_hard_set.py, add artifacts to submit_backup_recovery_artifacts.sh."
