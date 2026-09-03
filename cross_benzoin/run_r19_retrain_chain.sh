#!/bin/bash
# r1-9 retrain chain -- run once cross_round{8,9}_dft_sp.csv exist (i.e. after
# assemble_r89_dft_sp.py --rounds 8 9). CPU-only, ~1h total; the GNN retrain is a
# separate sbatch printed at the end.
#
# Recipe = round10 HANDOFF ZH §4.8-4.11, specialised to rounds 1-9 (no round10).
# Frozen 260-feature champion schema (unchanged since round7) lives at
#   cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json
#
#   bash cross_benzoin/run_r19_retrain_chain.sh
#
# step [2]'s relabel input candidates_v3/aldehydes_with_scaffold_split.parquet was a
# purge stub; REBUILT 2026-09-03 by cross_benzoin/rebuild_aldehydes_with_scaffold_split.py
# (220,524 rows, 80/10/10). Re-run that script if it ever goes missing again.
# Run this chain with PY=/home/schen3/venv/nequip/bin/python (has pyarrow/sklearn/xgboost).
#
set -euo pipefail
REPO="/gpfs/scratch1/shared/schen3/benzoin-dg-restored"
PY="${PY:-/home/schen3/venv/nhc-workflow/bin/python}"
NEQUIP_PY="/home/schen3/venv/nequip/bin/python"
cd "$REPO"

R9=data/cross_benzoin/cross_round9
SCHEMA=data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json

for r in 8 9; do
  f="data/raw/dft_sp_cross/cross_round${r}/cross_round${r}_dft_sp.csv"
  [[ -s "$f" ]] || { echo "MISSING $f -- run assemble_r89_dft_sp.py first"; exit 1; }
  echo "  $f: $(( $(wc -l < "$f") - 1 )) labelled pairs"
done

echo "=== [1/4] assemble r1-9 mordred table ==="
"$PY" cross_benzoin/assemble_cross_training_table_v3.py --rounds 1 2 3 4 5 6 7 8 9 \
  --product-mordred-csv \
    data/cross_benzoin/cross_round3/rounds123_products_mordred.csv \
    data/cross_benzoin/cross_round4/round4_products_mordred.csv \
    data/cross_benzoin/screen10k/screen10k_products_mordred.csv \
    data/cross_benzoin/cross_round8/round8_products_mordred.csv \
    data/cross_benzoin/cross_round9/round9_products_mordred.csv
# -> $R9/cross_train_table_9rounds_mordred.parquet

echo "=== [2/4] relabel scaffold-disjoint split ==="
"$PY" cross_benzoin/relabel_scaffold_split_9rounds.py
# -> $R9/cross_train_table_9rounds_scaffold_split_labeled.parquet

echo "=== [3/4] prune to frozen 260-feature champion schema ==="
"$PY" cross_benzoin/prune_table_to_champion_features.py \
  --table "$R9/cross_train_table_9rounds_scaffold_split_labeled.parquet" \
  --feature-list "$SCHEMA" \
  --out "$R9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet"

echo "=== [3b] NaN guard: every champion feature must be non-all-NaN on clean-train ==="
"$PY" - <<'PY'
import json, pandas as pd
R9 = "data/cross_benzoin/cross_round9"
fl = json.load(open("data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json"))
df = pd.read_parquet(f"{R9}/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet")
tr = df[df["new_scaffold_split"] == "train"] if "new_scaffold_split" in df else df
bad = [c for c in fl if c in tr and tr[c].isna().all()]
print(f"clean-train rows: {len(tr)} ; all-NaN champion feats: {len(bad)}")
if bad:
    print("  ABORT -- would crash MLPRegressor:", bad[:10]); raise SystemExit(1)
print("  OK")
PY

echo "=== [4/4] retrain champion + ensemble (scaffold-disjoint) ==="
"$PY" cross_benzoin/train_scaffold_disjoint.py \
  --table "$R9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet" \
  --outdir "$R9/scaffold_disjoint_9rounds_v1"

echo
echo "=== DONE steps 1-4. Now submit the r1-9 attentive GNN retrain (CPU/fat_rome): ==="
cat <<EOF
sbatch --job-name=gnn_attn_9r_cpu --partition=fat_rome --nodes=1 --ntasks=1 \\
  --cpus-per-task=32 --mem=120G --time=10:00:00 \\
  --output=$REPO/slurm_logs/gnn_attn_9r_cpu_%j.out \\
  --wrap='source /etc/profile; module load 2023; \\
    export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32; \\
    cd $REPO && $NEQUIP_PY -u cross_benzoin/train_cross_gnn_arch_sweep.py \\
      --table $R9/cross_train_table_9rounds_scaffold_split_labeled_slim260.parquet \\
      --champion-dir $R9/scaffold_disjoint_9rounds_v1 \\
      --ensemble-path $R9/scaffold_disjoint_9rounds_v1/models/ensemble_scaffold_disjoint.joblib \\
      --outdir $R9/gnn_attentive_9rounds_v1 \\
      --arch attentive --hidden 128 --layers 4 --lr 3e-4 --seed 0'
EOF
