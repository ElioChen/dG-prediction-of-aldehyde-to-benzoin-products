#!/bin/bash
#SBATCH --job-name=homo_sa_full
#SBATCH --partition=fat_genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/homo_sa_full_%j.out
#
# Full-library homo standalone tabular leg (CAMPAIGN_PLAN.md Phase 2+3):
# merge SP relabel shards -> assemble full ~184k table (champion 260-feat
# schema, dG_r2scan_kcal renamed to dG_orca_kcal by the assembler) -> single
# XGBoost Delta-model with B97-3c baseline (CB_BASELINE_COL=dG_b973c_kcal,
# CB_TARGET_COL left at its dG_orca_kcal default -- see assemble_homo_standalone_table.py
# build_full_library()). Run once homo SP archived+regen tracks both drain to
# 100% (HANDOFF_20260916 Sec1.3). GNN leg is the dependent job
# submit_homo_standalone_gnn_full.sh.
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
export CB_BASELINE_COL=dG_b973c_kcal

OUT=data/cross_benzoin/homo_standalone
SP=$OUT/relabel_sp
TABLE=$OUT/homo_standalone_full_library_table_slim260.parquet
echo "homo_sa_full node=${SLURMD_NODENAME} $(date)"

echo "-- Phase 2: merge SP relabel shards --"
$PY -u cross_benzoin/merge_homo_sp.py --out-prefix "$SP/homo_sp" || exit 1
cat "$SP/homo_sp_summary.json"

echo "-- Phase 2: assemble full-library table --"
if [[ ! -f "$TABLE" ]]; then
    $PY -u cross_benzoin/assemble_homo_standalone_table.py --full-library \
        --labels "$SP/homo_sp_labels.csv" --out "$TABLE" || exit 1
else
    echo "table exists, skip assemble: $TABLE"
fi

echo "-- Phase 3: single-XGB champion (CB_BASELINE_COL=$CB_BASELINE_COL) --"
$PY -u cross_benzoin/train_scaffold_disjoint.py \
    --table "$TABLE" --outdir "$OUT/tabular_full"
RC=$?
echo "Done $(date) exit=$RC"
exit $RC
