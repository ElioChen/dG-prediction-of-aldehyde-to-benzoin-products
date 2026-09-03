#!/bin/bash
#SBATCH --job-name=recov_backup
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#
# Back up everything the 2026-09-02 recovery produced, to home4 -- the only storage
# that survived the 2026-07 scratch purge (see the sibling
# submit_backup_xyz_geometry.sh and /gpfs/home4/schen3/benzoin_backups/).
#
# The purge destroyed data that was BOTH gitignored AND unbacked-up. Everything
# archived here is in exactly that category: too large for git, expensive to
# regenerate. Two of the three lost aldehyde library files were restorable only
# because an earlier session happened to tar them to home; this script makes that
# habit systematic rather than lucky.
#
# Idempotent: re-run at any time, tarballs are overwritten in place. Safe to run
# while arrays are still going -- it archives whatever exists at that moment.
#
# Submit:
#   sbatch cross_benzoin/slurm/submit_backup_recovery_artifacts.sh
#
REPO="${REPO:-/gpfs/scratch1/shared/schen3/benzoin-dg-restored}"
DEST="/gpfs/home4/schen3/benzoin_backups/recovery_20260902"
mkdir -p "$DEST"
cd "$REPO" || exit 1

echo "recov_backup node=${SLURMD_NODENAME:-unknown} $(date)"
echo "repo=$REPO dest=$DEST"

arc () {  # arc <tarball-name> <paths...>
    local name="$1"; shift
    local present=()
    for p in "$@"; do [[ -e "$p" ]] && present+=("$p"); done
    if [[ ${#present[@]} -eq 0 ]]; then
        echo "--- $name: nothing present yet, skipped"; return
    fi
    echo "--- $name: archiving ${#present[@]} path(s)"
    tar -czf "$DEST/${name}.tar.gz" "${present[@]}" 2>/dev/null
    ls -lh "$DEST/${name}.tar.gz"
}

# 1. the rebuilt/restored aldehyde library -- the three files whose loss blocked
#    everything. aldehydes_all.csv is the only one with no prior archive.
arc recovered_aldehyde_library \
    data/cross_benzoin/homo_v6/aldehydes_all.csv \
    data/cross_benzoin/homo_v6/aldehydes_mordred_slim102.csv \
    data/cross_benzoin/homo_v6/aldehydes_bdfe_gxtb_descriptors.csv \
    data/cross_benzoin/homo_v6/bdfe_gxtb_aldehydes

# 2. DFT-SP labels reconstructed from the tracked cv_predictions dumps. Small, but
#    these represent DFT compute that cannot be redone cheaply for rounds 1-7.
arc reconstructed_dft_sp_labels data/raw/dft_sp_cross

# 3. round10 stage1 derived tables (products/aldehydes merges, product mordred,
#    product g-xTB BDE). Descriptors only -- geometry goes in its own tarball.
arc cross_round10_fat20_stage1_descriptors \
    data/cross_benzoin/cross_round10_fat20_stage1/cross_round10_products_merged.csv \
    data/cross_benzoin/cross_round10_fat20_stage1/cross_round10_aldehydes_merged.csv \
    data/cross_benzoin/cross_round10_fat20_stage1/mordred_products \
    data/cross_benzoin/cross_round10_fat20_stage1/bde_gxtb

# 4. round10 stage1 3D geometry. The Jul-21 archive's cross_round10_xyz_geometry.tar.gz
#    is the ABANDONED earlier round10 attempt, not this one -- hence the distinct name.
#    Product mordred can only ever be recomputed from these.
echo "--- cross_round10_fat20_stage1_xyz_geometry ---"
XYZ=$(find data/cross_benzoin/cross_round10_fat20_stage1 -maxdepth 2 \
        \( -name xyz_prod -o -name xyz_ald \) -type d 2>/dev/null)
if [[ -n "$XYZ" ]]; then
    echo "  $(find $XYZ -name '*.xyz' 2>/dev/null | wc -l) .xyz files in $(echo "$XYZ" | wc -l) dirs"
    tar -czf "$DEST/cross_round10_fat20_stage1_xyz_geometry.tar.gz" $XYZ
    ls -lh "$DEST/cross_round10_fat20_stage1_xyz_geometry.tar.gz"
else
    echo "  no xyz dirs found, skipped"
fi

# 5. the aldehyde recompute (cb_featurize --aldehydes-only) -- descriptors + geometry.
echo "--- aldehyde_recovery_r17 ---"
if [[ -d data/cross_benzoin/aldehyde_recovery_r17 ]]; then
    tar -czf "$DEST/aldehyde_recovery_r17.tar.gz" \
        data/cross_benzoin/aldehyde_recovery_r17/recovery_pairs.csv \
        data/cross_benzoin/aldehyde_recovery_r17/todo_aldehydes.csv \
        data/cross_benzoin/aldehyde_recovery_r17/chunk_*/aldehydes.csv \
        data/cross_benzoin/aldehyde_recovery_r17/chunk_*/xyz_ald 2>/dev/null
    ls -lh "$DEST/aldehyde_recovery_r17.tar.gz"
else
    echo "  not present, skipped"
fi

# 5b. rounds 8-9 DFT-label recovery (2026-09-03). The product r2SCAN-3c SP chunks
#     under data/raw/dft_sp_cross/cross_round{8,9}/sp_products/ are already inside
#     tarball #2, but these thermal/geom sidecars + the regenerated aldehyde
#     funnel_v3 geometries (SLURM 26351006) are not, and the aldehyde geom+thermal
#     is real xTB compute lost in the purge with no prior backup.
arc r89_dft_recovery_sidecars \
    data/cross_benzoin/cross_round8_recover \
    data/cross_benzoin/cross_round9_recover \
    data/cross_benzoin/r89_aldehyde_recover/aldehyde_geom_list.csv \
    data/cross_benzoin/r89_aldehyde_recover/aldehyde_thermal.csv
echo "--- r89_aldehyde_regen_geometry ---"
if [[ -d data/cross_benzoin/r89_aldehyde_regen ]]; then
    tar -czf "$DEST/r89_aldehyde_regen_geometry.tar.gz" \
        data/cross_benzoin/r89_aldehyde_regen/r89_aldehyde_todo.csv \
        data/cross_benzoin/r89_aldehyde_regen/chunk_*/aldehydes.csv \
        data/cross_benzoin/r89_aldehyde_regen/chunk_*/xyz_ald 2>/dev/null
    ls -lh "$DEST/r89_aldehyde_regen_geometry.tar.gz"
else
    echo "  not present, skipped"
fi

# 6. the rebuilt rounds1-7 training table + reproduction retrain (the recovery's
#    validation result: MAE 1.877 vs historical 1.883, see
#    [[rounds17-reproduction-confirmed]] in Claude memory). The full unpruned table
#    is skipped (585 cols, superseded by the 260-feature slim one actually used).
arc rounds17_recovered_table_and_model \
    data/cross_benzoin/cross_round7/cross_train_table_7rounds_recovered_slim260.parquet \
    data/cross_benzoin/cross_round7/train_ensemble_7rounds_recovered_v1

echo
echo "Done $(date)"
echo "--- archive listing ---"
ls -lh "$DEST"
df -h /gpfs/home4/schen3 2>/dev/null | tail -1
