# screen_v6 — how well does a LINEAR model predict benzoin ΔG? (2026-06-17)

## Question
Given the xTB/morféus descriptors in `screen_v6`, how good is a plain **linear
model** at predicting the xTB benzoin self-condensation ΔG (`dG_xtb_kcal`)? This
sets the interpretable baseline that any nonlinear model (trees/GNN) must beat,
and quantifies how much of ΔG is a simple additive function of the descriptors.
Companion to `findings_2026-06-17_screen_v6_analysis.md` (the descriptor↔ΔG
correlation study).

## Method
- **Target** `dG_xtb_kcal` on the clean physical window [−40, 20] (the 239
  `|ΔG|>40` xTB failures excluded).
- **Features** 24 xTB+morféus descriptors (electronic + steric). The 7 Multiwfn
  ADCH/QTAIM columns are excluded — still being backfilled and empty for most rows.
- **Model** ordinary least squares (numpy `lstsq`), z-scored features, **5-fold
  cross-validation**, metrics on the held-out (out-of-sample) predictions.
- Run per grouping (all / aromatic in-scope / non-aromatic), since the drivers
  differ between regimes (see the correlation findings). Script: `linear_model.py`,
  `linear_parity_rest.py`.

## Results (out-of-sample, 5-fold CV)
| group | n | CV R² | RMSE (kcal) | MAE (kcal) | best single feat (R²) |
|---|---|---|---|---|---|
| all aldehydes | 219,504 | 0.443 | 3.72 | 2.63 | Mulliken q(O) = 0.37 |
| **aromatic in-scope** | 146,573 | **0.549** | **3.31** | **2.35** | Mulliken q(O) = 0.50 |
| non-aromatic (aliphatic) | 72,931 | 0.368 | 3.95 | 2.82 | Mulliken q(O) = 0.16 |

Parity plots (held-out predictions, one per group):
[aromatic](../../data/raw/screen_v6/analysis/fig15_linear_parity_aromatic.png) ·
[all](../../data/raw/screen_v6/analysis/fig16_linear_parity_all.png) ·
[non-aromatic](../../data/raw/screen_v6/analysis/fig17_linear_parity_nonaromatic.png).

`train R² ≈ CV R²` to <0.001 in every group → **no overfitting**; the numbers are
the genuine ceiling of the linear form, not a sampling artifact.

## Interpretation — linear is only moderate
- **Best case (aromatic) R²≈0.55, RMSE≈3.3, MAE≈2.35 kcal.** Versus the
  predict-the-mean baseline (σ≈4.9 kcal for aromatic), the model cuts error by only
  ~⅓ and leaves **~45% of the variance unexplained**.
- **It is essentially a one-descriptor story.** Mulliken q(O) alone gives R²=0.50;
  adding the other 23 descriptors lifts R² to just 0.55 (**+0.05**). The
  descriptor→ΔG map has real **nonlinearity / interactions** that an additive linear
  model cannot absorb.
- **The parity clouds show regression-to-the-mean**: the model compresses the range
  and systematically under-predicts the most exergonic substrates (actual −30 →
  predicted ≈ −15). The tails — exactly the candidates of interest — are where the
  linear model is worst.
- **Group ranking mirrors the correlation study.** Aromatic (electronic regime,
  q(O)/WBO/LUMO strongly linear) fits best; aliphatic (steric regime, weaker and
  more nonlinear) worst (R²=0.37, single-feature R² only 0.16); pooling the two
  dilutes to 0.44. Concrete support for **modelling aromatic separately**
  (`aromatic-only-scope`).
- RMSE ≈ 3.3 kcal coincides numerically with the ~3 kcal xTB-vs-DFT noise floor
  (`delta-mae-noise-floor`), **but the target here is xTB ΔG itself**, so this is
  not a noise limit — it is real structure the linear form misses, and a nonlinear
  model can push it lower.

## Caveat — don't read the OLS coefficients as importances
The standardized aromatic coefficients blow up on the Fukui block (fukui_0 +57.9,
f⁺ −29, dual −23, f⁻ −22). These are **multicollinearity artifacts**:
`fukui_0=(f⁺+f⁻)/2` and `dual=f⁺−f⁻` are linear combinations of f⁺/f⁻, so OLS
distributes huge cancelling weights among them. This does **not** affect predictive
R² (lstsq handles rank deficiency) but means the coefficients are not
interpretable. For descriptor importance use the correlation ranking instead
(q(O), C=O WBO, LUMO, ω). A ridge / de-correlated refit would stabilise them.

## Takeaways
1. A linear model is a fine **fast, interpretable coarse filter / sanity baseline**
   (R²≈0.55, MAE≈2.4 kcal for aromatics) but **not for fine ranking** — it is worst
   on the exergonic tail you care about.
2. The +0.05 gain from 24 features over 1 says **go nonlinear** (gradient-boosted
   trees / GNN) to capture the remaining ~45% variance — consistent with
   `gnn-delta-result` and `delta-mae-noise-floor`.
3. Model **per regime**: aromatic 0.55 > pooled 0.44; aliphatic needs its own
   steric-weighted treatment if ever revisited.

Figures use new filenames (`fig15/16/17`); no existing outputs overwritten.

## Simple neural network — does it beat linear? (yes)
A 2-hidden-layer MLP `(128, 64)` ReLU / Adam, z-scored features+target, early
stopping, **same 80/20 train/test split** as a refit linear baseline. sklearn
`MLPRegressor` in `venv/nhc-workflow`; script `nn_model.py`.

| group | model | test R² | RMSE | MAE | train R² (overfit gap) |
|---|---|---|---|---|---|
| aromatic | linear | 0.552 | 3.29 | 2.34 | — |
| aromatic | **MLP (128,64)** | **0.661** | **2.86** | **2.02** | 0.688 (+0.027) |
| all | linear | 0.437 | 3.73 | 2.63 | — |
| all | **MLP (128,64)** | **0.598** | **3.15** | **2.21** | 0.641 (+0.043) |

Parity (held-out): [aromatic](../../data/raw/screen_v6/analysis/fig18_nn_parity_aromatic.png) ·
[all](../../data/raw/screen_v6/analysis/fig18_nn_parity_all.png).

- The MLP lifts aromatic R² **0.55→0.66** and cuts MAE **2.34→2.02 kcal** (~14%);
  for the pooled set **0.44→0.60**. Converged in 47–65 iters with a tiny train–test
  gap (+0.03–0.04) → well-regularised, not overfit.
- This **confirms the nonlinearity**: the variance linear missed is partly real
  structure an MLP captures. The larger pooled gain (+0.16) shows the NN can model
  the aromatic-electronic and aliphatic-steric regimes jointly, where linear was
  diluted.
- MAE ≈2.0 kcal is surrogate fidelity to xTB ΔG (target is xTB itself), not a noise
  limit — still ~34% variance left. Headroom: add the 7 ADCH/QTAIM columns once
  backfilled (24→31 feats), tune/deepen, or try gradient-boosted trees / a GNN.

Next option: de-collinearised linear refit for clean coefficients; tree/GNN
comparison; re-run with ADCH/QTAIM features after the Multiwfn backfill finishes.

## Gradient-boosted trees vs tuned MLP (proper 70/20/10 split)
Stricter protocol: **train 70% fits, val 20% selects, test 10% touched once.**
Subsampled to 60k rows/group for the search (stable for ranking). GBT =
`HistGradientBoostingRegressor` (defaults + early stop, **no grid**); MLP grid =
{(128,64),(256,128),(256,128,64)} × alpha {1e-4,1e-3}, best picked on val.
Script `gbt_mlp_tune.py`.

| group | model | test R² | RMSE | MAE |
|---|---|---|---|---|
| aromatic | linear | 0.557 | — | — |
| aromatic | MLP* (256,128) | 0.651 | 2.91 | 2.05 |
| aromatic | **GBT** | **0.663** | 2.86 | 2.01 |
| all | linear | 0.449 | — | — |
| all | MLP* (256,128,64) | 0.572 | 3.30 | 2.31 |
| all | **GBT** | **0.585** | 3.25 | 2.27 |

Parity (test): GBT
[aromatic](../../data/raw/screen_v6/analysis/fig19_gbt_parity_aromatic.png) /
[all](../../data/raw/screen_v6/analysis/fig19_gbt_parity_all.png); tuned-MLP
[aromatic](../../data/raw/screen_v6/analysis/fig20_mlptuned_parity_aromatic.png) /
[all](../../data/raw/screen_v6/analysis/fig20_mlptuned_parity_all.png).

- **GBT is the practical winner** — best R², needs no feature scaling and no tuning
  (defaults + early stopping), edging the tuned MLP by ~0.01.
- **Both nonlinear models land at R²≈0.66 (aromatic) / ≈0.58 (all)** — the same
  ceiling. Deepening/widening the MLP did NOT push past the simple net; capacity is
  not the bottleneck.
- So **~0.66 is the nonlinear ceiling extractable from these 24 xTB descriptors**
  for aromatic: nonlinearity buys ~+0.10 R² over linear (0.56→0.66), but the
  remaining ~34% of ΔG variance is **not a function of this feature set** — it needs
  more information (the ADCH/QTAIM columns being backfilled, or full 3D structure →
  a GNN), not a bigger MLP.

### Ranking summary (held-out, aromatic)
linear 0.56  →  MLP/GBT ≈ 0.66. GBT recommended as the tabular baseline going
forward. Subsample/login-node caveat: search ran on 60k rows; the full-data simple
MLP earlier hit 0.66 too, so the ceiling is robust.

## Do the backfilled ADCH/QTAIM features help? (marginally)
After the Multiwfn backfill completed (1105/1105 chunks, ADCH ~98% / QTAIM ~99.9%
filled; merged to `features_mwf.csv`, aggregated to
`analysis/screen_v6_features_mwf_all.csv`), GBT was rerun with **24 vs 31 features
on the SAME rows/split** (60k subsample, 70/20/10; HistGBR handles the ~2% ADCH
NaNs natively). Script `mwf_feature_test.py`.

| group | features | test R² | RMSE | MAE |
|---|---|---|---|---|
| aromatic | 24 base | 0.626 | 3.04 | 2.10 |
| aromatic | **31 (+ADCH/QTAIM)** | **0.640** | 2.98 | 2.06 |
| all | 24 base | 0.588 | 3.20 | 2.25 |
| all | **31 (+ADCH/QTAIM)** | **0.597** | 3.17 | 2.21 |

- Adding the 7 Multiwfn descriptors gives a **small but clean gain** (+0.014
  aromatic, +0.009 all; same rows + split, only the feature columns differ).
- It does **not break the ~0.64–0.66 ceiling** — the ADCH/QTAIM block carries only
  ~1% extra explained variance over the 24 xTB/morféus descriptors. (Note absolute
  GBT R² shifts ±0.02–0.03 between random 60k subsamples — e.g. 0.626 here vs 0.663
  earlier — so only the *within-pair* delta is meaningful.)
- **Implication:** the remaining ~35% variance is not unlocked by more scalar
  descriptors; it needs a richer representation (3D structure → GNN, cf.
  `gnn-delta-result`). ADCH/QTAIM are worth keeping (consistent small lift,
  `descriptor-slim-v4`) but are not a game-changer for this target.

## Honest generalization: scaffold/cluster split (not just random)
Random splits leak similar molecules across train/test and can inflate R². Re-ran
GBT (31 feats, **full aromatic n=146,741**) under Bemis–Murcko **scaffold splits**
where whole scaffolds (21,638 unique; largest = benzene, 18%) go entirely to one
side — no scaffold leakage. Script `scaffold_split_eval.py`,
parity [fig21](../../data/raw/screen_v6/analysis/fig21_gbt_scaffold_parity_aromatic.png).

| split | n_test | test R² | RMSE | MAE |
|---|---|---|---|---|
| random-molecule (leaky reference) | 14,674 | 0.690 | 2.76 | 1.93 |
| scaffold-random (representative novel scaffolds) | 14,737 | **0.680** | 2.96 | 2.13 |
| scaffold-rare-test (hardest: unusual/rare chemotypes) | 14,675 | **0.542** | 3.68 | 2.58 |

- **Random splitting was only mildly optimistic for typical new molecules:**
  scaffold-random drops just **−0.01** (0.690→0.680). The model generalizes to
  unseen scaffolds almost as well as to random holdouts.
- **Why so robust:** it is built on physics-based descriptors (xTB electronics +
  sterics), which transfer across ring systems — it learns ΔG physics, not scaffold
  memorization. A fingerprint/structure model would have leaked far more.
- **But genuinely novel/rare chemotypes are harder:** predicting the rarest
  scaffolds (held entirely out of training) drops to **R²=0.54, MAE 2.6 kcal**
  (−0.15 vs random). Those exotic polycyclics/heterocycles sit in
  under-sampled descriptor regions → true feature-space extrapolation.
- **Practical guidance:** trust ~0.68 (MAE ~2.1) when screening the bulk library of
  ordinary substituted aromatics; expect ~0.54 (MAE ~2.6) and extra caution when
  prioritizing unusual/novel scaffolds. The headline 0.66 ceiling is NOT a
  random-split artifact.
