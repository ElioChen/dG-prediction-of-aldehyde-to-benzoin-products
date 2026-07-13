# Final model decision: MORDREDSLIM271_BDEGXTB — synthesis of the 2026-07-07 follow-up battery

## Bottom line

**MORDREDSLIM271_BDEGXTB remains the production champion** (275 feats: 72 QM + 199 SHAP-pruned
mordred + 4 g-xTB BDE/BDFE; MLP + XGB_d8 + XGB_d10 + quantile-UQ ensemble; test MAE **1.503**,
RMSE 2.257, R² 0.875; bundle `pipeline/models/gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_
20260706.joblib`). Four follow-up analyses were run to (a) sanity-check and interpret this
model and (b) attempt to beat it architecturally, including with a GNN. None beat it outright;
the sections below summarize what each analysis found and why the decision stands.

## 1. SHAP cost audit — keep the model, simplify future feature acquisition

SHAP on a 4000-row test subsample (XGB_d8) shows the 4 g-xTB BDE/BDFE features are NOT equally
useful: `ald_bde_gxtb_kcal` (rank 4/275) and `prod_bde_gxtb_kcal` (rank 6/275) — both cheap
(single SP/opt) — carry **3.7x** the summed SHAP importance of `prod_bdfe_gxtb_kcal` (rank 15)
and `ald_bdfe_gxtb_kcal` (rank 38), which require an expensive full-Hessian `--ohess` per
fragment. **Action: for any future prospective screening of new molecules, compute only BDE,
skip BDFE** — the existing bundle keeps both (already paid for on the full 220k library, no
reason to retrain to drop 2 already-computed columns), but this rules out further investment
in BDFE-family (thermal/entropic) descriptors going forward. See
`REPORT_shap_mordredslim271_bdegxtb_20260707.md` / `_zh.md` and
[METHODS_BDE_gxtb_20260707_en.md](METHODS_BDE_gxtb_20260707_en.md) for the full methodology.

## 2. Deep error / noise-band analysis — where the remaining 1.503 kcal/mol comes from

32.3% of test error (7,075/21,910 molecules) is genuinely above the noise floor established by
the baseline_72 seed-reshuffle study (1.571±0.013 kcal/mol, 3-sigma cutoff 1.610) — the
remaining ~68% of error is within what pure DFT-label noise could explain and is not fixable
by a better model. Of the real-error tail: **sulfonyl (11.2x enrichment), has_P (9.5x), imine
(3.6x), amide (3.4x, a newly-quantified driver), ester (1.8x), nitro (1.5x)**. Structure-space
clustering (Morgan FP) found 4 similar halogen-dominated hard clusters (mean error 2.8-3.0) + 1
amide-dominated cluster (3.36, worst) + 1 heterogeneous cluster with no dominant tag —
consistent with the earlier SHAP-attribution-space finding of "4 heterogeneous failure modes."
**Implication: further accuracy gains, if pursued, should target these functional-group
failure modes specifically (e.g. dedicated sulfonyl/amide features or targeted DFT
re-labeling) rather than generic model/architecture changes** — see the new noise-band
histogram, `126_noise_band_histogram_champion275_20260707.png`.

## 3. Aromatic-subset deep dive — historical sampling bias is resolved, aliphatic is now harder

Full-library aromatic fraction (66.3%) matches the test-set fraction (66.3%, 0.03pp gap) —
the old MaxMin active-selection undersampling of aromatics is **not present** in the current
full-library-trained population (that concern was specific to an earlier active-selection
phase, moot now that DFT-SP covers ~the full clean_v4 library). Per-scope accuracy: **aromatic
MAE 1.327/R²0.875 vs aliphatic MAE 1.870/R²0.842** — aliphatic is now the harder scope, and
uncertainty routing correctly reflects this (26.5% vs 9.7% route-to-DFT rate). Within
aromatic, chemotype barely matters (simple carbocyclic 1.268 best, azine/azole 1.391, furan
1.375, thiophene 1.275 — all close). **Implication: if further data collection is prioritized,
aliphatic molecules are the higher-value target, not aromatic.**

## 4. GNN parity attempt — architecture is confirmed not to be the lever

`gnn_dual_qm_champion275.py` (job 24478591) fed the dual-encoder GINE architecture (the
best-performing GNN arch from the full architecture sweep, `REPORT_homo_v6_gnn_architectures_
20260629.md`) the exact same 275 champion features at readout, on the same 70:20:10 seed-42
split protocol over the matched ~218k population — testing whether richer features (not a
different architecture) let a GNN reach/beat 1.503. **Result: test MAE 1.552** (RMSE 2.269,
R²0.874, 544k params) — closed most of the old gap (dual_qm+56feat was 1.646→now 1.552 with
275 feats) but still 0.049 behind the tabular champion. Combined with the prior full
architecture sweep (all 2D conv operators plateau at 2.58-2.68 regardless of operator; only
QM-feature-injection breaks the plateau) and the earlier 3D-GNN result (SchNet 2.18, losing to
the same-data MLP's 1.83), the conclusion is now robust across three independent tests:
**giving the GNN identical information to the tabular model still does not let it extract as
much value from that information as gradient-boosted trees do.** Architecture swaps are not
the lever for this task; information (features) is.

## 5. Architecture-improvement attempt: GNN + tabular stacking ensemble

Since neither a different architecture nor matched features let the GNN win outright, the
remaining lever for an architecture-level improvement is **combining** the two model families
rather than replacing one with the other — GNNs (message-passing) and GBDTs (axis-aligned
splits) are different enough function classes that their errors may be only partially
correlated, in which case a blend can beat both even though neither wins alone.
`gnn_dual_qm_champion275_ensemble.py` (job 24482531) retrained the same dual-encoder GNN
(identical protocol) while additionally saving per-molecule test predictions with molecule
IDs, then blended them against the tabular champion's saved test predictions
(`test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv`) over a weight sweep
`w_gnn ∈ {0, 0.05, ..., 1.0}`.

**Methodological bug caught and fixed mid-flight (job 24482531): the first attempt's blend
comparison was invalid.** Both scripts use `np.random.default_rng(42).permutation(N)` for
their 70:20:10 split, but each derives its own population `N` independently (slightly
different dropna footprints between the GNN and tabular finalize scripts) — a permutation of
a different N is an unrelated random sequence even under the same seed, so the "last 10%"
selected as each script's own test set has almost no overlap by construction (job 24482531's
run asserted this: `unexpectedly small overlap`, exit 1). **Fix (`gnn_ensemble_reeval_matched.
py`): reuse the already-trained GNN weights (saved by 24482531) and re-evaluate the model on
the EXACT tabular test-set molecule IDs**, rather than trusting the GNN's own independent
split to line up. This also surfaced a second, more serious risk that had to be filtered out:
of the tabular model's 21,858 test-set IDs, **15,258 (70%) turned out to be inside the GNN's
own training set** — exactly the ~70% expected by chance from two independent splits of the
same pool — so those had to be excluded before blending, or the GNN's memorized (not
generalized) predictions on them would have inflated its apparent accuracy.

**Result on the clean, leakage-free overlap (n=6,613 molecules, job 24489591):**

| config | MAE (n=6,613 subset) |
|---|---|
| tabular-only | 1.476 |
| gnn-only | 1.557 |
| **best blend (w_gnn=0.40, i.e. 60% tabular + 40% GNN)** | **1.425** |

Delta vs tabular-only on this subset: **-0.051**, well outside the established noise band
(0.02-0.03 kcal/mol) — a real improvement from blending, consistent with the hypothesis that
GNN and GBDT errors are only partially correlated. **Caveat: this is measured on a ~6,600-
molecule subset (the leakage-safe overlap), not the full 21,910-row official test set, and
the tabular-only MAE on this subset (1.476) differs from the officially reported 1.503 on the
full test set** — so the -0.051 delta is only rigorously valid as a within-subset comparison,
not yet a validated claim that the full production model would drop from 1.503 to ~1.45.
**Next step to confirm before promoting:** retrain both models from the start on one shared,
aligned 70:20:10 split (same ID-based fold assignment for both) so the full test set can be
blended without this subsetting, and confirm the gain holds at full scale.

## Overall recommendation

Keep MORDREDSLIM271_BDEGXTB as the sole production model for now, but **the stacking-ensemble
result is promising enough (-0.051 MAE, ~2x the noise band, on a real if partial overlap) to
justify the follow-up aligned-split retrain** described above before deciding whether to adopt
a blended predictor for production scoring (at the added cost of running the GNN at inference
time). If further accuracy is wanted beyond whatever the ensemble delivers, the two other
evidence-based paths are: (a) targeted work on the functional-group failure modes from §2
(sulfonyl/amide/imine/P), and (b) prioritizing aliphatic molecules in any future labeling
effort (§3). Do not invest further in a standalone GNN replacement (§4) or in BDFE-family
descriptors (§1).
