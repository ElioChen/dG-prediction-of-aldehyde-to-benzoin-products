# cross-benzoin ΔG champion

**Benchmark champion (2026-09-14): r1-10 B97-3c Δ-model blend, holdout MAE
0.528** (vs the deployed model's 2.215 below) — see §"r1-10-b973c" further
down. **Deployed/production default is still r1-10 (g-xTB baseline, MAE
2.215)**, described first in this file: `predict_dg.py` only knows how to
compute a g-xTB baseline for a brand-new pair, not a B97-3c one, so the
b973c model isn't drop-in yet. Wire that up before switching the deployed
default.

## r1-10 (g-xTB baseline) -- deployed default, adopted 2026-09-04

## Load

```python
from predict_cross_champion import CrossBenzoinBlendPredictor
pred = CrossBenzoinBlendPredictor.load(
    "data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_v1",
    gnn_dir="data/cross_benzoin/cross_round10/gnn_attentive_10rounds_v1",
)   # blend_w_gnn = 0.50, read from the GNN metadata.json
```

`predict_cross_champion.py` now defaults `--model-dir` / `--gnn-dir` to these.

## Components

| part | path |
|---|---|
| MLP+XGB ensemble | `cross_round10/scaffold_disjoint_10rounds_v1/models/ensemble_scaffold_disjoint.joblib` |
| single-XGB champion | `.../models/champion_scaffold_disjoint.joblib` |
| attentive-pooling GNN | `cross_round10/gnn_attentive_10rounds_v1/models/gnn_state.pt` |
| frozen 260-feature schema | `cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json` (unchanged since round 7) |
| training table | `cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet` (35,528 rows, clean-train 22,771) |

## Scaffold-disjoint holdout (n=448, frozen since round 7)

| | MAE | R² |
|---|---|---|
| **blend (w_gnn=0.50)** | **2.215** | — |
| MLP+XGB ensemble | 2.326 | 0.739 |
| GNN-only | 2.313 | — |
| single-XGB | 2.548 | 0.702 |
| g-xTB baseline | 5.037 | −0.14 |

Bootstrap (B=20000): blend−ensemble delta 0.111, 90% CI [0.047, 0.181] (excludes 0),
**P(blend better than ensemble) = 0.9986**.

## Why r1-10 over r1-9

r1-9 blend was 2.167 on the same holdout — nominally better, but within noise
(each MAE's bootstrap SE ≈ 0.15 at n=448). r1-10 is trained on 1,969 more DFT-SP
pairs (the round10 active-learning batch, which targets high-uncertainty regions the
frozen holdout doesn't sample), and its GNN blend advantage is if anything stronger
(w_gnn 0.40 → 0.50, delta 0.086 → 0.111). Latest + most data + robust blend wins;
the flat holdout is not a reason to keep the smaller model. **r1-9
(`cross_round9/scaffold_disjoint_9rounds_v1` + `gnn_attentive_9rounds_v1`) is the
fallback.**

## Lineage

r1-7 recovered (2026-09, post-purge; GNN null, w_gnn=0) → r1-9 (2026-09-04; r8/9
DFT labels recovered from scratch; **GNN becomes useful, w_gnn=0.40, P=0.995**) →
**r1-10** (2026-09-04; + round10 AL batch; w_gnn=0.50, P=0.999).
Full log: `RUN_LOG_20260903.md`.

## Predicting ΔG for new aldehyde pairs

**⚠ 2026-09-06→07**: the 09-06 BDE library rebuild (`bde_post_sweep`) silently broke this
tool for every new pair (a shared-file coverage-sentinel bug, see
`[[predict-dg-g-gxtb-regression-fixed]]` in Claude memory / `RUN_LOG_20260903.md` 09-07).
**Fixed** commit `c520acf`, re-smoke-tested 09-07 on 20 real round10 pairs — runs to
completion, predictions in the right order of magnitude. The original "MAE 1.65 on 5
known round10 pairs" validation below predates the regression; treat it as historical,
not current, until a fresh precision check is run (the 09-07 smoke test only confirmed
the pipeline *runs*, matching against true labels needs the right per-round DFT-SP file).

**Assemble+predict half** (if you already have a `cross_round*_dft_products.csv`-schema
file — id, donor_id, acceptor_id, donor_smiles, acceptor_smiles, smiles, xtb_*,
dG_gxtb_kcal, …):

```
/home/schen3/venv/nequip/bin/python cross_benzoin/predict_dg.py \
    --products-csv <pairs_products.csv> --out preds.csv
```
→ `preds.csv`: id, …, `dG_pred_kcal`, `ens_member_sigma` (cheap 3-learner spread,
not the full bootstrap epistemic estimate), plus three Goal-3 reformulation columns
added 2026-09-07 that are NOT capped by the ~2.9 kcal label-noise floor (validated AUC
0.92-0.93, see `eval_reformulation_classification_ranking.py`): `dg_favorable` (dG<0),
`dg_below_train_median`, `dg_rank_pct` (within-batch percentile, use for screening/
ranking a candidate batch, not a single pair).

**Deployment-hardening columns added 2026-09-07** (need `predict_dg_calibration.json`,
built by `build_predict_dg_calibration.py`; omitted if absent):
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

**From scratch** (new pairs, needs the slow GFN2-xTB geometry — newly wired, not yet
tested on genuinely novel pairs):

```
sbatch cross_benzoin/slurm/submit_predict_dg.sh <pairs.csv> <workdir> <out.csv>
# pairs.csv: donor_id,acceptor_id,donor_smiles,acceptor_smiles
```

Aldehydes must be in `data/library` (donor_*/acceptor_* descriptors are pulled from
the 220k library by canonical SMILES); `bde_gxtb_kcal` is pulled from any prior
round's `bde_gxtb/` for known products, NaN (median-filled) otherwise.

## r1-10-b973c -- new holdout-best model, 2026-09-14, not yet deployed

Tier B (`data/cross_benzoin/rec1_b973c_tierB/`) self-consistently relabeled
the same 35,528-pair r1-10 training set with r2SCAN-3c target + **B97-3c
baseline** (instead of g-xTB), completing 2026-09-14 (35,136/35,528 usable,
98.9% coverage, GREEN QC — B97-3c residual std 0.916 vs g-xTB's 3.785, ratio
0.242, matching the pilot's 0.26 at full scale, see
[[rec1_b973c_baseline_lever_validated]]). Retraining the champion pipeline
on this table (schema v2 = 257 features, drop `{donor,acceptor,product}_n_CHO`)
is a **breakthrough, not an incremental gain**:

| | MAE (n=448, same frozen holdout) | vs g-xTB-baseline champion (2.215) |
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

**Why this is not (yet) the deployed champion:** a B97-3c single point is a
real DFT calculation (cheap relative to a full geometry optimization, but
not free like g-xTB) that has to be computed for the product of every new
donor/acceptor pair before this model's baseline column is available.
`predict_dg.py`'s from-scratch and assemble+predict paths both only compute
`dG_gxtb_kcal`. Until that's wired up (add a B97-3c SP step alongside the
existing g-xTB one), r1-10-b973c is a benchmark/offline result, not a
drop-in replacement for `predict_cross_champion.py` / `predict_dg.py`.

Components: `data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_b973c_v1/`
(ensemble + single-XGB + 257-feature schema), `gnn_attentive_10rounds_b973c_seed{1..4}/`,
training table `cross_train_table_10rounds_scaffold_split_labeled_slim257_b973c.parquet`,
sweep result `cross_round10/gnn_seed_ensemble_r10_b973c_result.json`. Full recipe:
`data/cross_benzoin/rec1_b973c_tierB/DRAIN_RUNBOOK.md`.

**Next step to actually deploy this:** add a B97-3c SP compute step to
`predict_dg.py`'s new-pair path (mirrors the existing g-xTB step), re-run the
09-07 smoke-test precision check against it, then swap this file's "deployed
default" pointer.
