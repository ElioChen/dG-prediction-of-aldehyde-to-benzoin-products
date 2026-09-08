#!/bin/bash
#SBATCH --job-name=rec2_homo_pmordred
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/rec2_homo_pmordred_%j.out
#
# Rec-2 prep (Tier-B-parallel, label-independent). Two stages on one node:
#   1. extract archived homo_unify product geometries from
#      bde_homo_product_featurize_20260902/chunk_*/geom.tar.zst  (~2135 archives)
#   2. targeted product Mordred (add_mordred_cross_products.py families) over the
#      ~24.6k extracted geometries, 32-way process pool
# -> data/cross_benzoin/homo_unify/product_geoms/products_mordred_descriptors.csv
#    (also copied to homo_v6/products_mordred_descriptors.csv for the assembler)
# Short CPU job -> genoa short borrow is acceptable for this size. No ORCA/GPU.
set -euo pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
cd "$REPO"
source /etc/profile 2>/dev/null || true
OUT=data/cross_benzoin/homo_unify/product_geoms
mkdir -p "$OUT" slurm_logs
echo "rec2_homo_pmordred node=${SLURMD_NODENAME} $(date)"

# stage 1: extract
$PY -u cross_benzoin/rec2_extract_homo_product_geoms.py --out "$OUT"

# stage 2: chunked mordred, CHUNK=200, 32-way pool
PC="$OUT/homo_unify_products_for_mordred.csv"
NROWS=$($PY -c "import pandas as pd;print((pd.read_csv('$PC')['error'].astype('string').fillna('')=='').sum())")
NCHUNK=$(( (NROWS + 199) / 200 ))
echo "mordred: $NROWS geoms -> $NCHUNK chunks of 200"
seq 0 $((NCHUNK - 1)) | xargs -P 32 -I {} \
  $PY -u cross_benzoin/add_mordred_cross_products.py \
      --products-csv "$PC" --chunk-id {} --chunk-size 200 --out-dir "$OUT/mordred"

# concat
$PY -u - <<PYEOF
import pandas as pd, glob
fs = sorted(glob.glob("$OUT/mordred/chunk_*.csv"))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("id")
d.to_csv("$OUT/products_mordred_descriptors.csv", index=False)
d.to_csv("data/cross_benzoin/homo_v6/products_mordred_descriptors.csv", index=False)
val = int(d.drop(columns=["id"]).notna().any(axis=1).sum())
print(f"concat {len(d)} rows, {val} with any mordred value -> products_mordred_descriptors.csv")
PYEOF
echo "Done $(date)"
