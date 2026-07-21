# Cross-benzoin pilot: DFT-SP validation of the g-xTB baseline (2026-07-14)

Companion file: `REPORT_cross_pilot_dft_sp_validation_20260714_ZH.md` (中文版).

## What this job was

SLURM job **24609263** (`pipeline/slurm/submit_dft_sp_cross.sh`, genoa, 48 cores,
2h48m) ran r2SCAN-3c single-point DFT (`pipeline/compute/dft_sp_cross_from_geom.py`)
on the product geometries of the **598-row cross-benzoin pilot** (`cross_pilot_v1`,
300 unordered / 598 directed donor≠acceptor pairs, stratified across all 6
aromatic_carbo/aromatic_hetero/aliphatic category combinations — see
[[codex-cross-benzoin-session-20260714]]). The aldehyde side needed **zero new
compute**: both reactants of every pair were already at r2SCAN-3c in the
219k-row homo campaign. Only the 598 new product geometries got a fresh SP.

This is the validation gate flagged as the top open item in
`REPORT_codex_gap_analysis_20260714_EN.md`: *"once the pilot validates
(non-trivial ΔG spread, correct AB≠BA regiochemistry, low error rate), scale to
the diversity/uncertainty-driven few-thousand-row set."* All three criteria are
now checked against real DFT, not just g-xTB.

## Result 1 — zero failures

598/598 rows scored, **zero errors** (empty `error` column throughout). No
timeouts, no SCF convergence failures, no geometry rejects. This is the
cleanest completion rate of any DFT-SP campaign run in this project so far
(the homo funnel_v3 campaign had a documented ~4% ORCA timeout rate at the
same 3600s-class limit — see [[dftsp-timeout-3600-snapshot]] — this run used
a longer 7200s timeout and much smaller batch, so the comparison isn't
apples-to-apples, but it's a good sign for the cross product geometries
specifically: no new failure mode from having two different donor/acceptor
scaffolds in one product).

## Result 2 — non-trivial, sane ΔG spread

`dG_orca_kcal` (true DFT, r2SCAN-3c/CPCM-DMSO): mean 5.06, std 5.03,
range [−10.76, +24.29] kcal/mol, n=598. This is a real distribution, not a
degenerate spike — comparable in shape to the homo-benzoin DFT label spread,
and shifted positive (less exergonic) relative to the homo mean, which is
chemically plausible: cross pairs mix donor/acceptor electronic character
that isn't optimized for benzoin coupling the way a matched homo pair can be.

## Result 3 — AB≠BA regiochemistry confirmed at the DFT level

All 299 unordered pairs have both directed rows present (598 = 299×2, one
pair dropped to a missing aldehyde-cache entry). The previous session only
confirmed orientation-sensitivity at the g-xTB level (mean Δ 2.64 kcal
across 300 pairs); this job repeats the check on **true DFT energies**:

| stat | \|ΔG(A→B) − ΔG(B→A)\|, DFT (n=299 pairs) |
|---|---|
| mean | 3.70 kcal/mol |
| median | 3.07 kcal/mol |
| max | 15.47 kcal/mol |
| min | 0.02 kcal/mol |

Donor/acceptor direction is a first-order effect on ΔG at the reference
level of theory, not a g-xTB artifact. This directly supports the directed
(not symmetric-pair) modeling architecture in `NEXT_STEPS.md` Phase 3/5.

## Result 4 — g-xTB baseline generalizes to cross products, with the expected failure mode

This is the central quantitative result. Comparing both baselines to true DFT:

| model | MAE (kcal/mol) | RMSE | signed bias |
|---|---|---|---|
| raw GFN2-xTB (`dG_xtb_kcal`) | 15.68 | 16.05 | −15.68 (huge systematic underestimate) |
| **g-xTB (`dG_gxtb_kcal`)** | **3.45** | 4.57 | −2.33 |

Raw GFN2-xTB is, as expected, not usable in absolute terms (this matches the
established homo-benzoin finding that GFN2 alone needs a DFT/g-xTB correction
layer — [[gxtb-dft-correction-champion]]). g-xTB alone — with **no
cross-specific tuning, no BDE descriptors, no ML correction layer, and never
having seen a donor≠acceptor pair during any of its calibration** — lands at
MAE 3.45 kcal/mol. For context: this is close to (a bit above) this project's
established **~3.2 kcal/mol noise floor for the homo Δ-model**
([[delta-mae-noise-floor]]), and above the tuned homo champion (MAE 1.503,
1.427 with GNN stacking — [[gxtb-dft-correction-champion]],
[[gnn-stacking-confirmed-full-scale]]). In other words: **g-xTB's zero-shot
generalization to a chemically new regime (cross products) costs about
2 kcal/mol of MAE relative to the fully-tuned homo model, but is not broken.**
This is a good outcome — it means the existing g-xTB SP + the homo
Δ-model's feature/architecture choices are a sound starting point for cross
fine-tuning, not something that needs to be rebuilt from scratch.

Breakdown by category pair (g-xTB MAE, kcal/mol):

| category pair | n | MAE | median AE |
|---|---|---|---|
| aliph-carbo | 46 | 2.19 | 1.82 |
| aliph-hetero | 16 | 3.12 | 2.41 |
| carbo-hetero | 250 | 3.18 | 2.32 |
| carbo-carbo | 128 | 3.41 | 2.49 |
| aliph-aliph | 4 | 4.28 | 4.50 |
| hetero-hetero | 154 | 4.28 | 3.44 |

The ranking (aliphatic-involving pairs easiest, hetero-hetero hardest) is
**consistent with the project's established homo-benzoin failure mode**:
electron-withdrawing/heteroatom-rich substrates are where GFN2/g-xTB's
electronic-structure approximations break down hardest
([[nonewg-outlier-drivers]], [[screen-v6-funcgroup-analysis]]). The worst 8
outliers (|err| 12.3–15.8 kcal/mol) are all carbo-hetero, carbo-carbo, or
hetero-hetero pairs; none are aliphatic. This is the *same* known failure
mode reappearing in a new (cross) chemical space, not a new one — which is
reassuring for transferability of whatever correction strategy already works
on homo data (BDE/BDFE descriptors, EWG-aware features).

P90/P95 absolute error: 7.66 / 9.77 kcal/mol — a moderate tail, concentrated
in the hetero-heavy categories above.

## Bottom line: pilot validation gate — PASSED

| criterion (from the prior report) | result |
|---|---|
| non-trivial ΔG spread | ✅ mean 5.06, sd 5.03, range 35 kcal/mol wide |
| correct AB≠BA regiochemistry | ✅ confirmed at true DFT level, median 3.07 kcal/mol direction effect |
| low error rate | ✅ 0/598 (0%) |
| (bonus) g-xTB baseline usable pre-tuning | ✅ MAE 3.45, ~2 kcal above tuned homo champion, same failure mode as homo |

Every gate this pilot was designed to check has passed. The project is now
positioned to move to **Phase 3 (assemble the cross training table and train
the first tabular Δ-model / g-xTB baseline)** per `NEXT_STEPS.md`, using this
598-row set as the first labeled batch, or to first grow the labeled set via
another diversity/uncertainty round before training — see open decision below.

## Open decision (not resolved in this session)

`NEXT_STEPS.md` / `CROSS_BENZOIN_ML_RECOMMENDATIONS.md`'s active-learning loop
explicitly recommends training a first ensemble on a modest diverse subset,
then targeting high-uncertainty regions for the *next* labeling round, rather
than blindly growing the labeled set before ever training a model. 598 rows
is enough to fit a first g-xTB-correction baseline and get calibrated
uncertainty, but likely too small for a reliable molecule-disjoint holdout
test. Two reasonable paths, not yet chosen:

1. **Train now**: assemble the Phase 3 cross training table from these 598
   rows (role-aware donor/acceptor/product/delta descriptors per
   `DESCRIPTOR_POLICY_CROSS.md`), fit a first g-xTB-correction Δ-model, get
   per-row uncertainty, and use it to pick the next active-learning batch.
2. **Grow first**: run one more diversity-stratified pilot (a few thousand
   rows, still xTB/g-xTB only, no DFT) before spending any DFT budget on
   round 2, to have more rows once a train/val/test-disjoint split is cut.

## Addendum — decision made, first cross Δ-model trained (same session)

Given the go-ahead to proceed autonomously, path 1 ("train now") was chosen:
598 rows is enough for a first calibrated baseline, and this matches
`NEXT_STEPS.md`'s own active-learning loop more literally than growing the
label set blind. Two new scripts implement Phase 3 steps 1–2:

- `cross_benzoin/assemble_cross_training_table.py` — builds the role-aware
  table per `DESCRIPTOR_POLICY_CROSS.md`: `donor_*`/`acceptor_*` (aldehyde
  descriptors joined by **canonical SMILES**, not the `id` column —
  `aldehydes_all.csv`'s `id` is a row-index into `aldehydes_clean_v6.csv`,
  a different key than the `donor_id`/`acceptor_id` InChIKeys in the products
  table; this tripped the first run and was fixed before any model saw it),
  `product_*` (already role-aware from `cb_featurize.py`), RDKit-2D for all
  three roles, and an `interaction_*` block (HOMO(D)−LUMO(A) gap, Fukui
  match, steric/electronic mismatch terms). ADCH/QTAIM columns were all-null
  (this pilot never ran Multiwfn) and are dropped rather than imputed.
  Output: `data/cross_benzoin/cross_pilot_v1/cross_train_table.parquet`,
  598 rows × 147 features, 299 pairs.
- `cross_benzoin/train_cross_delta.py` — predicts the correction
  `dG_orca_kcal − dG_gxtb_kcal` with XGBoost (shallow: depth 3, 300 trees, to
  respect n=598). CV is **GroupKFold on the unordered pair key**, not plain
  K-fold — AB/BA share both parent molecules, so ungrouped CV would leak
  molecule identity across train/test. Repeated 5×20 to damp split noise at
  this size (same rationale as the homo model's repeated K-fold). Runs the
  full ablation set `DESCRIPTOR_POLICY_CROSS.md` requires.

**Result: MAE 2.09 (RMSE 2.70, R² 0.71), vs g-xTB baseline MAE 3.44** —
a **+1.35 kcal/mol** improvement from a first-pass model with no BDE/BDFE
descriptors, no GNN, and 598 training rows (two orders of magnitude smaller
than the 220k-row homo campaign). This is not yet at the tuned homo
champion's level (1.503/1.427), which is expected given the data-scale gap,
but it confirms the cross Δ-learning approach works at all, and — like the
g-xTB baseline itself — corrects hardest exactly where g-xTB was weakest:
hetero-hetero MAE drops from 4.28 → 2.39 and carbo-carbo from 3.41 → 1.79,
while aliph-carbo (already g-xTB's easiest category) drops only 2.19 → 1.73.

Ablation results (feature blocks, same CV protocol):

| block | n feats | MAE | RMSE | R² |
|---|---|---|---|---|
| 2D only | 48 | 2.91 | 3.71 | 0.454 |
| aldehydes only (donor+acceptor raw) | 52 | 2.70 | 3.49 | 0.517 |
| **product only** | 53 | **2.24** | 2.89 | 0.669 |
| donor+acceptor (raw+2D) | 100 | 2.72 | 3.48 | 0.520 |
| all raw blocks (no interaction terms) | 137 | 2.08 | 2.68 | 0.716 |
| all + interaction | 147 | 2.09 | 2.70 | 0.711 |

Two findings worth flagging for the next round: (1) **product-side
descriptors alone already recover most of the signal** (MAE 2.24 vs. full
2.08-2.09) — consistent with the homo-model finding that the low-level
physics layer at the reaction center carries most of the information, now
confirmed cross-domain; (2) **the hand-built interaction block did not
help at this n** (2.094 vs. 2.077 without it) — a small regression, likely
noise/mild overfitting from 10 extra correlated features on 598 rows rather
than a real negative signal; worth re-testing once the label set grows.
SHAP shows `P_int` (product global steric/electrostatic index) as by far
the top feature, with acceptor-side WBO/dipole/`P_int` next — physically
sensible, since the acceptor becomes the electrophilic carbinol center.

Artifacts: `data/cross_benzoin/cross_pilot_v1/train_v1/{models,figs,data}/`
(model joblib, feature list, metadata with full ablation table, parity plot,
per-category residual plot, SHAP importance plot, CV predictions CSV).

**Still open**: this is a single random-seed pair-grouped CV, not yet a
frozen, untouched molecule-disjoint test set (the doc's Phase 4 promotion
gate); this 598-row model is a first read to guide the next active-learning
batch, not a candidate for promotion. Next natural step per the active-
learning loop: use this model's per-row CV residual/uncertainty to target
the next labeling round at high-error regions (hetero-heavy pairs) and
under-represented categories (aliph-aliph, aliph-hetero — only 4 and 16
rows here).
