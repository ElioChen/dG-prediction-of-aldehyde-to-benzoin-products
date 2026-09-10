# homo-only ΔG model (standalone, from scratch)

Predict homo-benzoin ΔG on **homo data alone**, kept fully separate from the
cross pipeline. **From scratch on the full library** — the DFT labels lost in
the 2026-07 purge are recomputed, not worked around. Two **simple** models
(single XGBoost, single GNN), reported independently — not the iterated cross
champion blend.

Directive: user, 2026-09-10. Launch: after cross Tier B drains (~09-11).

## Plan

**`CAMPAIGN_PLAN.md`** — the full 3-phase plan.

1. **Relabel campaign** ("补算"): 161,630 pairs (2,000 QC + 159,630 to relabel),
   self-consistent geometry + r2SCAN-3c / B97-3c / g-xTB SP, 2 species/pair
   (`rec_homo_relabel_worker.py`). ~3 weeks. Smoke: job 26538271.
2. **Merge + assemble** (build while (1) runs): `merge_homo_relabel.py` +
   `assemble_homo_standalone_table.py --full-library --labels …` → full ~160k
   table, champion 260-feature schema, `dG_r2scan_kcal` target + `dG_b973c_kcal`
   Δ-baseline.
3. **Two simple models**: single-XGBoost Δ-model (`train_scaffold_disjoint.py`,
   report its single-XGB head) + single attentive GNN
   (`train_cross_gnn_arch_sweep.py`, seed 0, `gnn_only_mae`). Submit scripts
   `submit_homo_standalone.sh` (+`_gnn.sh`) already written — repoint `TABLE` at
   the full-library table + set `CB_TARGET_COL`/`CB_BASELINE_COL`.

## Data (2026-09-10)

Features present at full-library scale (aldehyde/product QM, Mordred restored
09-09, BDE, geometry — `homo_v6/LIBRARY_MANIFEST.md`). DFT ΔG labels: only
30,000 of ~219,000 survive → phase 1 recomputes.

- effective ceiling: **184,199** homo products with a valid structure; 24,569
  already carry a 30k label, **159,630 to relabel**
- full-library split `homo_v6/products_scaffold_split.csv`: train 175,636 /
  val 21,899 / test 21,886
- label `dG_orca_kcal` (surviving 30k): median 4.28, std 6.84 kcal/mol, heavy tails

## Results

_pending — phase 1 not yet launched (waits on Tier B drain)._

Benchmarks: cross champion blend MAE 2.215 (n=448); g-xTB baseline 5.037;
pre-purge full-library homo model ≈ MAE 10 (wider distribution).
