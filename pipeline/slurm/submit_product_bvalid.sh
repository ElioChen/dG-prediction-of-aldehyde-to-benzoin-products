#!/bin/bash
#SBATCH --job-name=prod_bvalid
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/slurm_logs/slurm-%A_%a.out
#
# B-validation: benzoin PRODUCT descriptors (xTB + morfeus, NO Multiwfn) on the 1695
# in-scope homo pairs, funnel_v3 geometry (matches training). One array task = one
# pre-split chunk input (featurize_product has no --skip). genoa bills 24-core units.
#
#   N=$(ls data/raw/product_bvalid/chunks_in/pairs_*.csv | wc -l)
#   sbatch --array=0-$((N-1))%64 submit_product_bvalid.sh
#
PROJ="/scratch-shared/schen3"; REPO="$PROJ/benzoin-dg"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SCRIPT="$REPO/pipeline/compute/featurize_product.py"
IN_DIR="$REPO/data/raw/product_bvalid/chunks_in"
OUT_DIR="$REPO/data/raw/product_bvalid/chunks_out"
XTB="/home/schen3/xtb/bin/xtb"
export OMP_NUM_THREADS=1
mkdir -p "$OUT_DIR"
source /etc/profile 2>/dev/null
TASK="${SLURM_ARRAY_TASK_ID:-0}"
IN="$IN_DIR/pairs_$(printf '%04d' "$TASK").csv"
OUT="$OUT_DIR/prod_$(printf '%04d' "$TASK").csv"
[[ -s "$OUT" ]] && { echo "chunk $TASK done — skip"; exit 0; }
# self-heal: clear this node's dead-job orphan scratch before generating our own.
# Live jobs are protected by clean_node_orphans' per-id scontrol check (NOT by a long
# mtime grace — a job-timeout-sized grace lets recent dead orphans fill /scratch-local
# and quota-kills the run). timeout 30 so a slow SLURM controller can't block job start.
timeout 30 bash "$REPO/pipeline/slurm/clean_node_orphans.sh" 2>/dev/null || true
SCRATCH="/scratch-local/$USER.$SLURM_JOB_ID.$TASK"; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
echo "task=$TASK in=$IN out=$OUT"
# funnel_v3 (default), NO --multiwfn (xTB+morfeus only -> 220k-feasible).
# CPU budget = workers * parallel_jobs * xtb_cores must = 24 (the allocation). Earlier
# 24*24 over-subscribed 576 procs / 24 cores -> memory thrash. 12*2*1 = 24, balanced.
# --xyz-dir = node-local scratch: reactant/product xyz are discarded with the job (no
# shared-FS inode bloat). funnel_v3 is deterministic → regenerate any geometry on demand.
# --xyz-dir = node-local (per-molecule xyz discarded); --xyz-merge-dir = shared, gets the
# 2 consolidated multi-frame xyz per chunk (reactant + product) for later use.
"$PY" "$SCRIPT" --input "$IN" --output "$OUT" --work-dir "$SCRATCH" --xyz-dir "$SCRATCH" \
      --xyz-merge-dir "$OUT_DIR/../xyz_merged" \
      --xtb-bin "$XTB" --conformer funnel_v3 --solvent dmso \
      --workers 12 --xtb-cores 1 --parallel-jobs 2
echo "done task=$TASK rc=$?"
