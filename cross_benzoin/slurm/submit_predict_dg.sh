#!/bin/bash
#SBATCH --job-name=predict_dg
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/predict_dg_%j.out
#
# Featurize-half of end-to-end ΔG prediction for NEW aldehyde pairs, then predict.
#   sbatch submit_predict_dg.sh <pairs.csv> <workdir> <out.csv> [--with-b973c]
# pairs.csv columns: donor_id,acceptor_id,donor_smiles,acceptor_smiles
# (donor_id/acceptor_id = InChIKeys; if you only have SMILES, precompute the keys.)
# --with-b973c (2026-09-14): also compute dG_b973c_kcal (needed for the r1-10-b973c
# champion, CHAMPION.md; real extra ORCA cost, off unless asked). After this script,
# call predict_dg.py with --baseline-col dG_b973c_kcal --model-dir .../
# scaffold_disjoint_10rounds_b973c_v1 --gnn-dir .../gnn_attentive_10rounds_b973c_seed<k>
# --schema .../scaffold_disjoint_10rounds_b973c_v1/models/feature_list.json.
#
# Bottleneck: cb_featurize's GFN2-xTB conformer search + ohess geometry, ~minutes/pair
# (parallel over --workers). Then a reshape to the cross_round*_dft_products.csv
# schema (+ dG_gxtb_kcal baseline, + dG_b973c_kcal if --with-b973c), then
# cross_benzoin/predict_dg.py (assemble+predict).
#
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
FEAT_PY=/home/schen3/venv/nhc-workflow/bin/python
NEQ_PY=/home/schen3/venv/nequip/bin/python
PAIRS="${1:?pairs.csv}"; WORK="${2:?workdir}"; OUT="${3:?out.csv}"
WITH_B973C=""
[[ "${4:-}" == "--with-b973c" ]] && WITH_B973C="--with-b973c"
mkdir -p "$WORK"
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export GXTB_BIN=/home/schen3/xtb/bin/xtb GXTB_SOLV="cosmo dmso" XTBPATH=/home/schen3/xtb/share/xtb
export ORCA_BIN=/home/schen3/orca/orca
echo "predict_dg featurize $(date)  pairs=$PAIRS work=$WORK with_b973c=${WITH_B973C:-no}"

# 1. cb_featurize: product QM/geometry + g-xTB SP (+ B97-3c SP if --with-b973c), and
#    emit the donor/acceptor aldehydes
"$FEAT_PY" cross_benzoin/cb_featurize.py --pairs "$PAIRS" --out "$WORK" --emit-aldehydes \
  $WITH_B973C \
  --xtb-bin /home/schen3/xtb/bin/xtb --solvent dmso --n-confs 10 --conformer funnel_v3 \
  --workers 20 --xtb-cores 2 --parallel-jobs 1

# 2. products.csv already IS cross_round*_dft_products.csv-shaped: featurize_pair()
#    writes id/donor_id/acceptor_id/donor_smiles/acceptor_smiles/smiles/reaction_type
#    AND the already-computed dG_xtb_kcal/dG_gxtb_kcal[/dG_b973c_kcal] columns directly
#    (see PROD_FIELDS in cb_featurize.py) -- no re-derivation needed. NOTE 2026-09-14:
#    this step used to re-join against aldehydes.csv to recompute dG_gxtb_kcal by hand,
#    reading a "features.csv" that cb_featurize.py has never written (it writes
#    "products.csv") -- the from-scratch predict_dg path was broken end-to-end since it
#    was wired (CHAMPION.md's "newly wired, not yet tested on genuinely novel pairs" was
#    accurate). Fixed by using the authoritative columns products.csv already has.
"$FEAT_PY" - "$WORK" <<'PY'
import sys, pandas as pd
work = sys.argv[1]
df = pd.read_csv(f"{work}/products.csv", low_memory=False)
df.to_csv(f"{work}/products_for_assemble.csv", index=False)
msg = f"staged {len(df)} pairs -> {work}/products_for_assemble.csv  (dG_gxtb non-null: {df['dG_gxtb_kcal'].notna().sum()}"
msg += f", dG_b973c non-null: {df['dG_b973c_kcal'].notna().sum()})" if "dG_b973c_kcal" in df.columns else ")"
print(msg)
PY

# 3. assemble + predict (g-xTB deployed default; pass --baseline-col/--model-dir/--gnn-dir/
#    --schema for the b973c model per the header comment above)
"$NEQ_PY" cross_benzoin/predict_dg.py --products-csv "$WORK/products_for_assemble.csv" --out "$OUT"
echo "Done $(date) -> $OUT"
