#!/bin/bash
#SBATCH --job-name=sameconf
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#
# Same-conformer, all-DMSO geometry-method test on a stratified 1% sample.
# Reuses the saved GFN2 benzoin geometries (conformer held fixed); per mol does
# r2SCAN-3c CPCM(DMSO) SP on the GFN2 geom + g-xTB --opt --cosmo dmso restart +
# r2SCAN-3c CPCM(DMSO) SP on the g-xTB geom. genoa bills 24-core units.
#
#   SEL=.../gxtb_test/sameconf_solv_sample.csv
#   N=$(($(wc -l < "$SEL")-1)); CH=20; NT=$(( (N+CH-1)/CH ))
#   sbatch --array=0-$((NT-1))%16 submit_sameconf_solv.sh
#
PROJ="/scratch-shared/schen3"
PY="/home/schen3/venv/nhc-workflow/bin/python"
SCRIPT="$PROJ/gxtb_test/gxtb_sameconf_solv.py"
SEL="$PROJ/gxtb_test/sameconf_solv_sample.csv"
XYZSRC="$PROJ/benzoin-dg/data/raw/screen_v6/dft_sp_r2scan3c/benzoin_xyz"
RESULTS="$PROJ/gxtb_test/sameconf_solv_chunks"
CHUNK="${CHUNK:-20}"
WORKERS="${WORKERS:-24}"
export OMP_NUM_THREADS=1
mkdir -p "$RESULTS"

source /etc/profile 2>/dev/null
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SKIP=$(( TASK * CHUNK ))
OUT="$RESULTS/chunk_$(printf '%05d' "$TASK").csv"
if [[ -s "$OUT" ]]; then echo "chunk $TASK done — skip"; exit 0; fi
timeout 30 bash "$PROJ/benzoin-dg/pipeline/slurm/clean_node_orphans.sh" 2>/dev/null || true  # self-heal: timeout protects against SLURM controller throttling
SCRATCH="/scratch-local/$USER.$SLURM_JOB_ID.$TASK"
mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
echo "task=$TASK skip=$SKIP chunk=$CHUNK out=$OUT"
"$PY" "$SCRIPT" --input "$SEL" --xyzsrc "$XYZSRC" --out "$OUT" \
      --scratch "$SCRATCH" --skip "$SKIP" --max "$CHUNK" --workers "$WORKERS"
echo "done task=$TASK rc=$?"
