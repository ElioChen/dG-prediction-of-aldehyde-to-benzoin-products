#!/bin/bash
#SBATCH --job-name=bde_geomarch
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/bde_geomarch_%j.out
#
# Incremental geometry archiver for the BDE homo/product featurize array (job 26316404).
#
# WHY: each finished chunk_XXXX/ holds ~200 tiny .xyz files under xyz/ + ald_xyz/.
# 2209 chunks => ~450k inodes, ~1.6 GB. The 2026-09-02 HANDOFF plan was to `rm -rf`
# them after assemble. The user asked to COMPRESS + KEEP the geometries instead
# (they are needed for the Phase-3 3D reaction-difference model, and rounds 8/9 losing
# their geometries was a real setback). This packs each sealed chunk's xyz dirs into a
# single per-chunk geom.tar.zst (verified) then removes the loose files, reclaiming
# ~200 inodes/chunk continuously DURING the run so scratch-inode usage does not climb.
#
# This is also the permanent replacement for HANDOFF step 2.2.2 (`rm -rf .../xyz`).
#
# Safety: a chunk is only touched when its featurization is provably complete
#   (features.csv row count >= pairs.csv row count) AND features.csv has not been
#   modified for >= SEAL_MIN minutes. A requeue of an already-complete chunk hits the
#   submit script's resume guard and `exit 0` without re-touching xyz/, so this is safe
#   even against a live array.
#
# Usage:
#   sbatch pipeline/bde/archive_completed_chunk_geoms.sh
# or run in foreground / background locally:
#   bash pipeline/bde/archive_completed_chunk_geoms.sh
set -o pipefail

REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
OUTDIR="${OUTDIR:-$REPO/data/cross_benzoin/bde_homo_product_featurize_20260902}"
ARRAY_JOB="${ARRAY_JOB:-26316404}"
SEAL_MIN="${SEAL_MIN:-30}"          # features.csv must be this many minutes stale
SLEEP_SEC="${SLEEP_SEC:-1800}"      # pause between sweeps
ZSTD_LEVEL="${ZSTD_LEVEL:-12}"
HOME_BK="${HOME_BK:-/home/schen3/benzoin_backups/bde_homo_rebuild_20260902}"
LOG_PREFIX="[geomarch $(date +%H:%M:%S)]"

mkdir -p "$OUTDIR/_geomarch_logs"
echo "$LOG_PREFIX start  REPO=$REPO OUTDIR=$OUTDIR ARRAY_JOB=$ARRAY_JOB SEAL_MIN=$SEAL_MIN SLEEP_SEC=$SLEEP_SEC"

archive_one() {
  local ch="$1"                      # absolute path to chunk_XXXX
  local tag; tag=$(basename "$ch")
  [[ -f "$ch/.geom_archived" ]] && return 0
  [[ -d "$ch/xyz" || -d "$ch/ald_xyz" ]] || { touch "$ch/.geom_archived"; return 0; }
  [[ -f "$ch/features.csv" && -f "$ch/pairs.csv" ]] || return 0

  local np nf
  np=$(($(wc -l < "$ch/pairs.csv") - 1))
  nf=$(($(wc -l < "$ch/features.csv") - 1))
  [[ "$np" -ge 1 ]] || return 0
  [[ "$nf" -ge "$np" ]] || return 0            # featurization not finished

  # staleness guard: features.csv untouched for >= SEAL_MIN minutes
  if [[ -n "$(find "$ch/features.csv" -mmin -"$SEAL_MIN")" ]]; then
    return 0
  fi

  local dirs=()
  [[ -d "$ch/xyz" ]] && dirs+=("xyz")
  [[ -d "$ch/ald_xyz" ]] && dirs+=("ald_xyz")
  local want
  want=$(find "${dirs[@]/#/$ch/}" -type f | wc -l)

  local tarf="$ch/geom.tar.zst"
  if ! tar -C "$ch" -cf - "${dirs[@]}" | zstd -q -T4 -"$ZSTD_LEVEL" -o "$tarf" -f; then
    echo "$LOG_PREFIX WARN $tag: tar|zstd failed, leaving loose files" >&2
    rm -f "$tarf"; return 1
  fi
  local got
  got=$(zstd -dc "$tarf" | tar -tf - | grep -c -E '\.xyz$')
  if [[ "$got" -ne "$want" ]]; then
    echo "$LOG_PREFIX WARN $tag: verify mismatch want=$want got=$got, keeping loose files" >&2
    rm -f "$tarf"; return 1
  fi
  rm -rf "${dirs[@]/#/$ch/}"
  touch "$ch/.geom_archived"
  echo "$LOG_PREFIX $tag: archived $got xyz -> $(du -h "$tarf" | cut -f1), freed ~$want inodes"
  return 0
}

sweep() {
  local n_arch=0 n_seen=0
  for ch in "$OUTDIR"/chunk_*; do
    [[ -d "$ch" ]] || continue
    n_seen=$((n_seen+1))
    if archive_one "$ch"; then
      [[ -f "$ch/.geom_archived" && ! -d "$ch/xyz" ]] && n_arch=$((n_arch+1))
    fi
  done
  local total_arch
  total_arch=$(find "$OUTDIR" -maxdepth 2 -name .geom_archived | wc -l)
  echo "$LOG_PREFIX sweep done: chunks=$n_seen archived_total=$total_arch"
}

job_alive() {
  squeue -j "$ARRAY_JOB" -h -o "%t" 2>/dev/null | grep -qE 'R|PD|CG'
}

while true; do
  LOG_PREFIX="[geomarch $(date +%H:%M:%S)]"
  sweep
  if ! job_alive; then
    echo "$LOG_PREFIX array $ARRAY_JOB no longer in queue -> final sweep then exit"
    sleep 120
    LOG_PREFIX="[geomarch $(date +%H:%M:%S)]"
    sweep
    # one consolidated copy of all per-chunk geom tarballs into home backup
    if [[ -d "$HOME_BK" ]]; then
      tar -C "$OUTDIR" -cf "$HOME_BK/homo_product_chunk_geoms_20260902.tar" \
        $(cd "$OUTDIR" && ls -d chunk_*/geom.tar.zst 2>/dev/null) 2>/dev/null \
        && echo "$LOG_PREFIX consolidated geom tarballs -> $HOME_BK/homo_product_chunk_geoms_20260902.tar"
    fi
    echo "$LOG_PREFIX exit"
    break
  fi
  sleep "$SLEEP_SEC"
done
