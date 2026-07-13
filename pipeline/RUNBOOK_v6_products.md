# Runbook — full screen_v6 unified featurization (after v6 screen finishes)

Trigger: SLURM array **23930563** (v6 xTB screen) fully COMPLETED (squeue empty).

Uses the clean **`cross_benzoin/`** package (see `cross_benzoin/ARCHITECTURE.md`):
aldehyde AND homo-benzoin product featurized in ONE funnel_v3 pass via
`cb_featurize.py --emit-aldehydes`, with Multiwfn, stable-ID naming, everything
saved. This supersedes the screen's plain-ranker aldehyde descriptors (the screen
stays the triage layer / `dG_xtb_kcal`) and the deprecated
`backfill_multiwfn`/`merge_multiwfn`. Scale = full 220,859 (confirmed).

```bash
REPO=/scratch-shared/schen3/benzoin-dg
SCREEN=$REPO/data/raw/screen_v6
OUT=$REPO/data/cross_benzoin/homo_v6
cd $REPO
```

## 1. Screen-level assemble + analysis (triage layer)
```bash
python pipeline/analyze_screen_v5.py --screen-dir $SCREEN
python pipeline/diagnose_screen.py  --screen-dir $SCREEN
```

## 2. Build homo pairs (donor==acceptor) with stable IDs
```bash
mkdir -p $OUT
python - "$SCREEN" "$OUT/homo_pairs.csv" <<'PY'
import csv, glob, sys
S, out = sys.argv[1], sys.argv[2]
w = csv.writer(open(out, "w")); w.writerow(["donor_id","acceptor_id","donor_smiles","acceptor_smiles"]); n=0
for f in sorted(glob.glob(f"{S}/chunk_*/features.csv")):
    for r in csv.DictReader(open(f)):
        if r.get("xtb_optimized")=="True" and r.get("SMILES"):
            i=r["index"]; w.writerow([i,i,r["SMILES"],r["SMILES"]]); n+=1
print("homo pairs:", n)
PY
```

## 3. Unified funnel_v3 array — aldehyde + homo product, NO Multiwfn, ALL saved
Per chunk writes `chunk_*/aldehydes.csv`, `chunk_*/products.csv`,
`chunk_*/xyz_ald/ald_<id>.xyz`, `chunk_*/xyz_prod/prod_<id>.xyz` (homo single-id).
```bash
IN=$OUT/homo_pairs.csv
N=$(($(wc -l < "$IN")-1)); CHUNK=100; NCH=$(( (N+CHUNK-1)/CHUNK )); mkdir -p $OUT/logs
# MULTIWFN=0 at full scale: L3 ADCH/QTAIM is ~min/molecule -> infeasible at 220k (those
# columns stay empty, filled only on subsets), the validated Δ-model path is Multiwfn-free,
# and it keeps the per-molecule scratch (inode) footprint low for %128. EMIT_ALD=1 is
# mandatory — product xyz + xTB/g-xTB G are needed for ΔG even if product descriptors don't
# help the ML (that's all B-validation decides). g-xTB baseline = separate SP step
# (pipeline/compute/gxtb_baseline.py) on these funnel_v3/GFN2-ohess geometries.
sbatch --array=0-$((NCH-1))%128 --output="$OUT/logs/cb_%A_%a.out" \
  --export=ALL,INPUT=$IN,OUTDIR=$OUT,CHUNK=$CHUNK,MULTIWFN=0,EMIT_ALD=1,CONFORMER=funnel_v3,N_CONFS=10 \
  cross_benzoin/slurm/submit_cb_featurize_array.sh
```

## 4. After the array finishes: concatenate tables
```bash
python - "$OUT" <<'PY'
import pandas as pd, glob, sys
O=sys.argv[1]
for name,tag in (("aldehydes","aldehydes_all"),("products","products_all")):
    fs=sorted(glob.glob(f"{O}/chunk_*/{name}.csv"))
    if not fs: continue
    df=pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if name=="aldehydes": df=df.drop_duplicates("id")          # homo: 1 ald per pair, but be safe
    df.to_csv(f"{O}/{tag}.csv", index=False); print(tag, len(df))
PY
```

Outputs (funnel_v3, method-consistent, structures saved, ID-linked):
- `$OUT/aldehydes_all.csv` + `$OUT/chunk_*/xyz_ald/`
- `$OUT/products_all.csv`  + `$OUT/chunk_*/xyz_prod/`

Notes: Multiwfn `/home/schen3/mutiwfn/Multiwfn_noGUI`. For future CROSS runs, feed a
cross pairs CSV to the same array (aldehydes will recur across chunks → dedup `id`
at concat, or pre-featurize aldehydes once). PILOT jobs 23936609/23936610 are a
preserved first cut (plain-ranker ald side), NOT the production reference.
```
```
