# cross-benzoin ΔG champion

**Current champion + deployed default (2026-09-14): r1-10 B97-3c Δ-model
blend, holdout MAE 0.528** (was r1-10 g-xTB, MAE 2.215 — see §"r1-10
(g-xTB)" below, kept as fallback). `predict_dg.py`/`predict_cross_champion.py`
now support both; defaults still point at the g-xTB model for
back-compat, so pass the b973c args explicitly (§"Load" below) until the
defaults themselves are flipped.

## r1-10-b973c -- champion, adopted 2026-09-14

Tier B (`data/cross_benzoin/rec1_b973c_tierB/`) self-consistently relabeled
the same 35,528-pair r1-10 training set with r2SCAN-3c target + **B97-3c
baseline** (instead of g-xTB), completing 2026-09-14 (35,136/35,528 usable,
98.9% coverage, GREEN QC — B97-3c residual std 0.916 vs g-xTB's 3.785, ratio
0.242, matching the pilot's 0.26 at full scale, see
[[rec1_b973c_baseline_lever_validated]]). Retraining the champion pipeline
on this table (schema v2 = 257 features, drop `{donor,acceptor,product}_n_CHO`)
was a **breakthrough, not an incremental gain**:

| | MAE (n=448, same frozen holdout) | vs old g-xTB champion (2.215) |
|---|---|---|
| **blend, 4-seed GNN (w_gnn=0.85)** | **0.528** | **-1.687 (-76%)** |
| single-XGB Δ-model | 0.632 | -1.583 |
| MLP+XGB ensemble | 0.603 | -1.613 |
| GNN, 4-seed average | 0.531 | -1.684 |
| GNN, best single seed | 0.556 | -1.659 |

Seed-averaging the GNN leg still helps here too (0.576 seed-1-only ->
0.531 at 4 seeds), same lever as [[gnn-seed-ensemble-lever]] on the old
champion, just far more headroom to work with now.

**Why the B97-3c baseline works this much better:** the Δ-model only has to
learn `r2SCAN-3c − B97-3c`, and that residual is already tight
(std 0.916 kcal, vs g-xTB's baseline residual std 3.785) *before any ML at
all* — B97-3c is simply a much closer starting point to the r2SCAN-3c label
than semiempirical g-xTB is, so the learned correction is smaller and easier.

**Deployment cost, now paid:** a B97-3c single point (real DFT, cheap
relative to a full geometry optimization but not free like g-xTB) now runs
for the product + donor + acceptor of every new pair, via
`cb_featurize.py --with-b973c` (2026-09-14). Real extra compute per pair,
but the wiring is done and validated (below), not a blocker anymore.

**Real-pair precision check (2026-09-14, not just the frozen holdout replay):**
3 known test-split pairs, re-run completely from scratch (fresh GFN2-xTB
conformer search + ohess + ORCA B97-3c/g-xTB SP, not read from the training
table) through the actual `submit_predict_dg.sh` -> `predict_dg.py` pipeline:

| pair | true dG | g-xTB pred | g-xTB err | b973c pred | b973c err |
|---|---|---|---|---|---|
| JJYUTVUHNPIWGF/GAOHPGKVCIOCHV | -3.689 | -2.376 | 1.31 | -4.072 | 0.38 |
| GAOHPGKVCIOCHV/JJYUTVUHNPIWGF | -3.844 | -0.162 | 3.68 | -3.543 | 0.30 |
| RSLLMRKKPRKJNB/XMTCKNXTTXDPJX | -6.456 | 2418.7 (!) | -- | -62.8 (!) | -- |

First two rows: b973c MAE 0.34 vs g-xTB MAE 2.50 -- both land right on
their respective holdout MAE (0.528 / 2.215), a genuinely independent
confirmation, not a training-table replay. Third row: **both models fail
badly on the same pair** (a zwitterion-prone amino-acid-like donor,
`C[C@@H]([C@H]([C@H]([C@@H](C=O)OC)O)OC)O` / `C(=O)[C@H](C(=O)O)N`) -- not
a b973c-specific weakness, a shared upstream feature/geometry issue on this
structure class. The g-xTB path's `dg_high_sigma` flag correctly caught it
(`ens_member_sigma=760`, way above the calib threshold); the b973c path has
no calibration yet (next gap, below) but the raw `ens_member_sigma=355` vs
the other two rows' 0.27/0.02 would have been an obvious tell even
unflagged. Not chased further -- know it's a real, caught failure mode, not
a silent one.

Components: `data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_b973c_v1/`
(ensemble + single-XGB + 257-feature schema), `gnn_attentive_10rounds_b973c_seed{1..4}/`,
training table `cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet`,
sweep result `cross_round10/gnn_seed_ensemble_r10_b973c_result.json`. Full
training recipe: `data/cross_benzoin/rec1_b973c_tierB/DRAIN_RUNBOOK.md`.

**Gap closed 2026-09-15:** `cross_benzoin/predict_dg_calibration_b973c.json`
built (`build_predict_dg_calibration.py`, now generalized with `--table
--model-dir --gnn-dir --blend-w-gnn --baseline-col --label-col` rather than
hardcoded to the g-xTB champion), calibrated against the **true 4-seed
champion** (see "corrected 2026-09-15" note above). `predict_dg.py`
auto-attaches it for `--baseline-col dG_b973c_kcal` via
`CALIB_JSON_BY_BASELINE` (any other/unregistered `--baseline-col` still
skips the columns explicitly, not silently). Numbers, n=929 pooled
test+validation: split-conformal 90% interval half-width **±1.21 kcal**
(vs g-xTB's ±5.24 kcal — 4.3x tighter, tracks the MAE gap), coverage
0.908 test / 0.894 validation. `baseline_risk` (same 7 SMARTS motifs) is
weaker evidence here than for g-xTB: flagged rows do have higher blend MAE
(0.80 vs 0.50) but **flat** `|baseline error|` (4.94 vs 4.98) — B97-3c
itself isn't failing on these structures, so treat the flag as "expect
worse accuracy," not "the cheap baseline is wrong, use DFT" (that framing
is still correct for the g-xTB model). `dg_high_sigma` threshold (p99 of
the 3-learner spread): 0.802 kcal, a different scale from g-xTB's own
threshold — each is read from its own JSON, not shared.

## Load

**⚠ corrected 2026-09-15**: the 0.528 headline MAE is the **4-seed GNN
average** (`blend_gnn_seed_ensemble_r10_b973c.py` sweep, `w_gnn=0.85`), not
any single seed dir. The `predict_cross_champion.py`/`predict_dg.py` code
written on 2026-09-14 only ever loaded `..._seed4` alone (`w_gnn=0.70` from
that seed's own `metadata.json`, test MAE 0.544) — a real, previously
undetected doc/code gap (0.544 deployed vs 0.528 documented), not a rounding
difference. `CrossBenzoinBlendPredictor.load()` now accepts a **list** of
`gnn_dir`s and averages their predictions (each seed's own norm stats used
for de-normalization; `blend_w_gnn` must be passed explicitly for a list —
guessing which sweep row applies would silently ship the wrong weight).
Re-verified: 4-seed load reproduces MAE 0.5284 on the frozen holdout,
matching the sweep's 0.5284 to 5 decimal places.

```python
from predict_cross_champion import CrossBenzoinBlendPredictor
pred = CrossBenzoinBlendPredictor.load(
    "data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_b973c_v1",
    gnn_dir=[f"data/cross_benzoin/cross_round10/gnn_attentive_10rounds_b973c_seed{s}" for s in (1, 2, 3, 4)],
    blend_w_gnn=0.85,
)
dg = pred.predict(df, baseline_col="dG_b973c_kcal")  # NOT the default dG_gxtb_kcal
```

`predict_dg.py` CLI: `--baseline-col dG_b973c_kcal --model-dir
data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_b973c_v1
--gnn-dir data/cross_benzoin/cross_round10/gnn_attentive_10rounds_b973c_seed1,...,seed4
--blend-w-gnn 0.85
--schema data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_b973c_v1/models/feature_list.json`
(comma-separated `--gnn-dir` = multi-seed average, `--blend-w-gnn` required with it).
`products-csv` must carry `dG_b973c_kcal` (from `cb_featurize.py --with-b973c`,
or already present in a Tier B-relabeled table).

Single-seed loading (`gnn_dir` as one path, no `blend_w_gnn`) still works
unchanged for back-compat / quick smoke tests, but is **not** the champion
config — MAE 0.544, not 0.528.

## Predicting ΔG for new aldehyde pairs

**Assemble+predict half** (if you already have a `cross_round*_dft_products.csv`-schema
file — id, donor_id, acceptor_id, donor_smiles, acceptor_smiles, smiles, xtb_*,
dG_gxtb_kcal[, dG_b973c_kcal], …):

```
/home/schen3/venv/nequip/bin/python cross_benzoin/predict_dg.py \
    --products-csv <pairs_products.csv> --out preds.csv
    # + --baseline-col/--model-dir/--gnn-dir/--schema for b973c, see "Load" above
```
→ `preds.csv`: id, …, `dG_pred_kcal`, `ens_member_sigma` (cheap 3-learner spread,
not the full bootstrap epistemic estimate), plus three Goal-3 reformulation columns
added 2026-09-07 that are NOT capped by the ~2.9 kcal label-noise floor (validated AUC
0.92-0.93, see `eval_reformulation_classification_ranking.py`): `dg_favorable` (dG<0),
`dg_below_train_median`, `dg_rank_pct` (within-batch percentile, use for screening/
ranking a candidate batch, not a single pair).

**Deployment-hardening columns** (g-xTB and b973c both covered as of
2026-09-15, see the b973c "Gap closed" note above; `CALIB_JSON_BY_BASELINE`
in `predict_dg.py` maps `--baseline-col` -> calibration JSON, built by
`build_predict_dg_calibration.py`; omitted for any other/unregistered
`--baseline-col`). Numbers below are the g-xTB champion's; see the b973c
section above for its own (tighter) numbers:
- `dG_pi_lo_90` / `dG_pi_hi_90` — split-conformal 90% prediction interval, calibrated
  on the scaffold-disjoint test+validation residuals (n=929). Distribution-free
  marginal coverage ≥ 90% (empirically 0.94 test / 0.87 validation). Half-width
  **±5.24 kcal** — wide because the point estimate sits on the label-noise floor and
  residuals are heavy-tailed. The σ-normalised variant was no tighter (σ carries too
  little conditional signal), so this is the plain global interval.
- `baseline_risk` (bool) / `baseline_risk_motifs` — the pair carries a
  g-xTB-baseline-failure substructure (hypervalent P / sulfonyl / sulfoxide / nitro /
  N-oxide / Se / triflate). On the holdout those rows have blend MAE **2.86 vs 2.10**
  and |g-xTB baseline error| **6.35 vs 4.81** (mirrors the homo hard-tail finding,
  corr(residual, baseline error) 0.888). Treat `baseline_risk=True` as "route to DFT".
- `dg_high_sigma` (bool) — `ens_member_sigma` above the calib p99 (2.81 kcal): the 3
  base learners wildly disagree (OOD structure). Ignore both `dG_pred_kcal` and the
  interval for these.

**From scratch** (genuinely new pairs, needs the slow GFN2-xTB geometry --
verified end-to-end 2026-09-14, both the g-xTB and b973c paths):

```
sbatch cross_benzoin/slurm/submit_predict_dg.sh <pairs.csv> <workdir> <out.csv> [--with-b973c]
# pairs.csv: donor_id,acceptor_id,donor_smiles,acceptor_smiles
# then re-run predict_dg.py by hand with the b973c args (see "Load") if --with-b973c
# was used -- the sbatch script itself still runs the g-xTB model by default.
```

**⚠ history, fixed 2026-09-14**: this from-scratch path had two bugs blocking
it since it was written -- `submit_predict_dg.sh` read a `features.csv`
`cb_featurize.py` never wrote (writes `products.csv`), and the 09-08 n_CHO
feature-audit drop silently broke assembling a table for ANY new pair
against the 260-feature schema (calc_rdkit still computes n_CHO; only the
assembler's column *selection* had dropped it — fixed by keeping it in the
assembled table, `include_ncho=True`). Both are fixed; the "real-pair
precision check" above is the first time this path has actually been
validated against ground truth, not just confirmed to run.

Aldehydes must be in `data/library` (donor_*/acceptor_* descriptors are pulled from
the 220k library by canonical SMILES); `bde_gxtb_kcal` is pulled from any prior
round's `bde_gxtb/` for known products, NaN (median-filled) otherwise.

## r1-10 (g-xTB baseline) -- fallback, adopted 2026-09-04, champion until 2026-09-14

### Load

```python
from predict_cross_champion import CrossBenzoinBlendPredictor
pred = CrossBenzoinBlendPredictor.load(
    "data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_v1",
    gnn_dir="data/cross_benzoin/cross_round10/gnn_attentive_10rounds_v1",
)   # blend_w_gnn = 0.50, read from the GNN metadata.json
```

`predict_cross_champion.py` / `predict_dg.py` still default `--model-dir` /
`--gnn-dir` / `--baseline-col` to these.

### Components

| part | path |
|---|---|
| MLP+XGB ensemble | `cross_round10/scaffold_disjoint_10rounds_v1/models/ensemble_scaffold_disjoint.joblib` |
| single-XGB champion | `.../models/champion_scaffold_disjoint.joblib` |
| attentive-pooling GNN | `cross_round10/gnn_attentive_10rounds_v1/models/gnn_state.pt` |
| frozen 260-feature schema | `cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json` (unchanged since round 7) |
| training table | `cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet` (35,528 rows, clean-train 22,771) |

### Scaffold-disjoint holdout (n=448, frozen since round 7)

| | MAE | R² |
|---|---|---|
| **blend (w_gnn=0.50)** | **2.215** | — |
| MLP+XGB ensemble | 2.326 | 0.739 |
| GNN-only | 2.313 | — |
| single-XGB | 2.548 | 0.702 |
| g-xTB baseline | 5.037 | −0.14 |

Bootstrap (B=20000): blend−ensemble delta 0.111, 90% CI [0.047, 0.181] (excludes 0),
**P(blend better than ensemble) = 0.9986**.

### Why r1-10 over r1-9

r1-9 blend was 2.167 on the same holdout — nominally better, but within noise
(each MAE's bootstrap SE ≈ 0.15 at n=448). r1-10 is trained on 1,969 more DFT-SP
pairs (the round10 active-learning batch, which targets high-uncertainty regions the
frozen holdout doesn't sample), and its GNN blend advantage is if anything stronger
(w_gnn 0.40 → 0.50, delta 0.086 → 0.111). Latest + most data + robust blend wins;
the flat holdout is not a reason to keep the smaller model. r1-9
(`cross_round9/scaffold_disjoint_9rounds_v1` + `gnn_attentive_9rounds_v1`) is the
next fallback down.

### Lineage

r1-7 recovered (2026-09, post-purge; GNN null, w_gnn=0) → r1-9 (2026-09-04; r8/9
DFT labels recovered from scratch; **GNN becomes useful, w_gnn=0.40, P=0.995**) →
**r1-10** (2026-09-04; + round10 AL batch; w_gnn=0.50, P=0.999) -> **r1-10-b973c**
(2026-09-14; B97-3c baseline swap, MAE 2.215 -> 0.528, new champion, see top of file).
Full log: `RUN_LOG_20260903.md`.
