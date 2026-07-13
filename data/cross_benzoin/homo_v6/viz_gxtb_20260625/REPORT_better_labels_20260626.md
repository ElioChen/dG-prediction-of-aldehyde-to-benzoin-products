# Can better labels break the ~1.6 kcal/mol MAE floor? (2026-06-26)

g-xTB→DFT ΔG correction. Two label-quality levers tested on held-out **test** molecules:
multi-conformer **Boltzmann-averaged DFT** and a **higher-level functional (wB97X-3c)**.
Jobs: confnoise 24225782 (product-side conformer spread, 34 mols) + boltzrelab 24226784
(full ΔE re-label, 120 mols, both aldehyde & product). Optuna 24225459 (regularized XGB) folded in.

## TL;DR
1. **Conformer averaging is a ~0.9 kcal/mol *removable* noise on the labels** (uncorrelated with model
   error, corr −0.07). Retraining on Boltzmann-averaged labels could lower the floor **~1.57 → ~1.38**
   (quadrature), at ~5× the DFT cost. The single biggest remaining lever — but modest and expensive.
2. **Functional choice moves labels MORE and *systematically*** (wB97X-3c − r2SCAN-3c: mean −0.49,
   std 1.69) and **correlates +0.31 with the model residual** — i.e. part of what looks like model
   "error" vs r2SCAN-3c is r2SCAN-3c functional error that wB97X-3c shifts toward the model. This is a
   *target-definition* question, not denoising: it cannot be claimed as a free floor reduction.
3. **Model side is exhausted**: Optuna-tuned regularized XGB = test **1.557** (vs default 1.59,
   ensemble 1.566). Tuning ≈ noise. Confirms the floor is data, not model.
4. **Caveat that dominates everything**: only 95/120 (and 71/95 for wB97X-3c) finished; survivors have
   mean |g-xTB Δ| **3.73** vs failed **7.96**. The hard EWG/large tail — which *sets* the floor —
   is under-sampled. These noise numbers describe the easy 79% (subset MAE 0.97 ≪ global 1.57).

## 1. Conformer label noise — two measurements, reconciled
| measurement | what it is | value |
|---|---|---|
| confnoise (24225782) | std of **product** r2SCAN-3c energy over 5 **random** conformers | mean **2.68**, median 2.28 kcal |
| boltzrelab boltz_corr | (lowest-xTB conf) − (xTB-Boltzmann avg) on **full ΔE** (ald+prod) | mean +0.11, **std 0.92**, mean\|·\| 0.62 |

The numbers differ because they measure different things. confnoise's 2.68 is the raw spread across
*random* conformers of *one* species — an upper bound. The production protocol uses the **lowest-xTB
conformer** (not random) of **both** species, and reactant/product conformer errors partially cancel in
ΔE = E_prod − 2·E_ald. The honest *protocol* noise is therefore **0.92 kcal/mol (boltz_corr)**, not 2.7.
boltz_corr⟂residual (corr −0.07) ⇒ it is genuine reducible noise.

## 2. Functional sensitivity (wB97X-3c vs r2SCAN-3c, same geometry)
mean −0.49, std 1.69, mean|·| 1.41 kcal/mol; corr(func_shift, model residual) = **+0.31** (n=71).
The mean offset is a systematic functional bias; the +0.31 correlation says the two functionals
disagree *in the direction of the model's apparent error*. So an unknown fraction of the 1.6 floor is
r2SCAN-3c's own functional error, not model/feature error — but we have no higher reference (DLPNO-
CCSD(T)) to say which functional is *right*, so this is not a bankable floor reduction, only evidence
the label definition is itself uncertain at ≥1 kcal.

## 3. Re-scoring the frozen model against each label set (95 test mols)
| label set | model MAE | Δ vs stored |
|---|---|---|
| stored single-conf r2SCAN-3c | **0.970** | — |
| Boltzmann-averaged r2SCAN-3c | 1.284 | +0.314 |
| wB97X-3c single-conf | 1.500 | +0.530 |
MAE *rises* under both — expected, since the model was trained on the single-conf r2SCAN target and the
new labels differ by (uncorrelated 0.9 / systematic 1.7) kcal. This does **not** contradict point 1:
the frozen-model swap measures label *movement*, while the floor reduction in point 1 assumes
*retraining* on the cleaner target (which removes the orthogonal 0.9 noise). Retraining needs a
full-library multi-conformer re-label, not done here.

## 4. Optuna regularized XGB (24225459)
best: max_depth 9, lr 0.051, min_child_weight 24, reg_lambda 31.9 → train 0.436 / val 1.553 / **test 1.557**
(default d10: train 0.56 / test 1.59). A single tuned XGB now ties the ensemble (1.566). Model-side
headroom is gone.

## Verdict & recommendation
The ~1.6 floor is **data-limited, not model-limited**. The only lever that could move it is a
**multi-conformer Boltzmann-averaged DFT re-label** (best case ~1.57→~1.38), and only after retraining;
expected gain ~0.2 kcal at ~5× DFT cost — **not recommended** unless sub-1.4 accuracy is required.
A higher functional changes the target rather than denoising and needs a CCSD(T) anchor to adjudicate.
Highest-value next step if pursued: re-label the **hard EWG/large tail** (the failed 21%, where the
floor actually lives) with multi-conformer DFT, not the easy bulk. See [[delta-mae-noise-floor]],
[[conformer-search-noise]], [[dft-labels-r2scan-not-pbe0]], [[gxtb-dft-correction-champion]].

Files (ts 20260626_1559): `boltz_relabel_results_*.csv`, `boltz_relabel_summary_*.md`,
`boltz_relabel_{corr,func}_hist_*.png`, `confnoise_results_*.csv`, `confnoise_summary_*.md`,
`confnoise_std_hist_*.png`.
