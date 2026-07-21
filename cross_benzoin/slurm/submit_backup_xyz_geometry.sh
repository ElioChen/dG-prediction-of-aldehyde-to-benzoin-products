#!/bin/bash
#SBATCH --job-name=xyz_backup
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#
# Back up the SURVIVING product/aldehyde 3D geometry (xyz_prod/, xyz_ald/) for
# cross-benzoin rounds 2,3,4,8,9 from scratch1 to home4, following the same tar.gz
# convention as the existing /gpfs/home4/schen3/benzoin_backups/homo_v6_scratch_archive/
# (descriptor-only backups from an earlier session). Motivated 2026-07-21: user asked
# whether 3D geometry was preserved anywhere -- audit found the 220k aldehyde library and
# all homo-track product geometry were deleted with NO backup (only descriptor CSVs
# survived), and rounds 1/5/6/7 of cross-benzoin never had xyz_prod persisted at all.
# Rounds 2/3/4/8/9 are the only cross-benzoin rounds with intact, validated geometry
# (round9 spot-checked 100% match/validation via gnn3d_extract_product_geometry.py) --
# this is the last line of defense against losing them the same way. Round10 (in
# progress as of this writing) gets backed up in a follow-up run once its featurize
# array completes.
#
# Real sustained I/O/CPU work (tar+gzip over ~41,500 files, several GB) -- goes through
# sbatch per memory no-login-node-compute.md, not run inline.
#
REPO="/scratch-shared/schen3/benzoin-dg"
DEST="/gpfs/home4/schen3/benzoin_backups/cross_benzoin_xyz_archive"
mkdir -p "$DEST"

cd "$REPO/data/cross_benzoin"
echo "xyz_backup node=${SLURMD_NODENAME} $(date)"
for r in cross_round2 cross_round3 cross_round4 cross_round8 cross_round9; do
    echo "--- $r ---"
    # xyz_prod (+ retry*/xyz_prod where present) and xyz_ald (small cache-miss set)
    DIRS=$(find "$r" -maxdepth 3 \( -iname "xyz_prod" -o -iname "xyz_ald" \) -type d 2>/dev/null)
    if [[ -z "$DIRS" ]]; then
        echo "  no xyz dirs found for $r, skipping"
        continue
    fi
    n_files=$(find $DIRS -iname "*.xyz" 2>/dev/null | wc -l)
    echo "  found $n_files .xyz files across: $(echo "$DIRS" | wc -l) dirs"
    tar -czf "$DEST/${r}_xyz_geometry.tar.gz" $DIRS
    ls -lh "$DEST/${r}_xyz_geometry.tar.gz"
done
echo "Done $(date)"
echo "--- final archive listing ---"
ls -lh "$DEST"
