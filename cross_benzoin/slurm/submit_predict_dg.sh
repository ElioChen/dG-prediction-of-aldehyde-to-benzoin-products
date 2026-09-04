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
#   sbatch submit_predict_dg.sh <pairs.csv> <workdir> <out.csv>
# pairs.csv columns: donor_id,acceptor_id,donor_smiles,acceptor_smiles
# (donor_id/acceptor_id = InChIKeys; if you only have SMILES, precompute the keys.)
#
# Bottleneck: cb_featurize's GFN2-xTB conformer search + ohess geometry, ~minutes/pair
# (parallel over --workers). Then a reshape to the cross_round*_dft_products.csv
# schema (+ dG_gxtb_kcal baseline), then cross_benzoin/predict_dg.py (assemble+predict).
#
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
FEAT_PY=/home/schen3/venv/nhc-workflow/bin/python
NEQ_PY=/home/schen3/venv/nequip/bin/python
PAIRS="${1:?pairs.csv}"; WORK="${2:?workdir}"; OUT="${3:?out.csv}"
mkdir -p "$WORK"
cd "$REPO"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export GXTB_BIN=/home/schen3/xtb/bin/xtb GXTB_SOLV="cosmo dmso" XTBPATH=/home/schen3/xtb/share/xtb
echo "predict_dg featurize $(date)  pairs=$PAIRS work=$WORK"

# 1. cb_featurize: product QM/geometry + g-xTB SP, and emit the donor/acceptor aldehydes
"$FEAT_PY" cross_benzoin/cb_featurize.py --pairs "$PAIRS" --out "$WORK" --emit-aldehydes \
  --xtb-bin /home/schen3/xtb/bin/xtb --solvent dmso --n-confs 10 --conformer funnel_v3 \
  --workers 20 --xtb-cores 2 --parallel-jobs 1

# 2. reshape features.csv (+ emitted aldehyde G_gxtb) -> cross_round*_dft_products.csv schema
"$FEAT_PY" - "$PAIRS" "$WORK" <<'PY'
import sys, pandas as pd
HARTREE = 627.5094740631
pairs = pd.read_csv(sys.argv[1]); work = sys.argv[2]
feat = pd.read_csv(f"{work}/features.csv", low_memory=False)
ald  = pd.read_csv(f"{work}/aldehydes.csv", low_memory=False)   # has smiles, G_gxtb, G_xtb
from rdkit import Chem
def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
gg = {canon(r.smiles): r.G_gxtb for r in ald.itertuples() if pd.notna(getattr(r, "G_gxtb", None))}
gx = {canon(r.smiles): r.G_xtb  for r in ald.itertuples() if pd.notna(getattr(r, "G_xtb",  None))}
df = feat.copy()
df["donor_smiles"]    = pairs["donor_smiles"].values
df["acceptor_smiles"] = pairs["acceptor_smiles"].values
df["donor_id"]    = pairs["donor_id"].values
df["acceptor_id"] = pairs["acceptor_id"].values
df["id"] = df["donor_id"].astype(str) + "__" + df["acceptor_id"].astype(str)
df["smiles"] = df["product_smiles"]
dcn = df["donor_smiles"].map(canon); acn = df["acceptor_smiles"].map(canon)
gpx = df.get("G_product")
df["dG_gxtb_kcal"] = (gpx - dcn.map(gg) - acn.map(gg)) * HARTREE
df["dG_xtb_kcal"]  = (df.get("xtb_energy") - dcn.map(gx) - acn.map(gx)) * HARTREE if "xtb_energy" in df else pd.NA
df.to_csv(f"{work}/products_for_assemble.csv", index=False)
print(f"reshaped {len(df)} pairs -> {work}/products_for_assemble.csv  "
      f"(dG_gxtb non-null: {df['dG_gxtb_kcal'].notna().sum()})")
PY

# 3. assemble + predict
"$NEQ_PY" cross_benzoin/predict_dg.py --products-csv "$WORK/products_for_assemble.csv" --out "$OUT"
echo "Done $(date) -> $OUT"
