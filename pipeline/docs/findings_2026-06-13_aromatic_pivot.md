# Benzoin ΔG — findings & decisions digest (2026-06-13)

Session arc: from "train tree+GNN on the new 1500-molecule set" → broad model
exploration → energy diagnostics → stricter cleaning → CHO-environment discovery →
**pivot to an aromatic-only model**. This is the condensed reference; see the named
scripts/figs for detail.

---

## 1. Model exploration — ~3.2 kcal MAE is a NOISE FLOOR
`explore_models.py` (n=1427, repeated-KFold CV of the Δ-correction → ΔG):

- **15 regression families all land MAE 3.23–3.44.** Plain **ridge ties the best
  (3.23)**; xgb/rf/gpr/lightgbm/stack all ≈3.24. No family beats the trees — even a
  linear model hits the floor.
- **Learning curve** (`learning_curve.png`): tree 3.43→3.21 over n=200→1427
  (diminishing); hybrid D-MPNN constant **+0.19 above** the tree (no crossover at 3×
  data). GNN stays the runner-up.
- **Uncertainty** (for active learning): **GPR best-calibrated** — PICP80 0.83,
  NLL 2.88 (vs ngboost 0.69 / quantile-GBM 0.54).
- **Classification reframe**: favourability (ΔG<0) classifier **AUC 0.846**.

**Why it's a floor:** the energy diagnostic (next) shows the xTB→DFT correction is a
near-constant offset + structure-independent noise. → Lever ranking: **label quality
> conformer/geometry noise > model choice ≈ more data.** Don't chase MAE via new
model families or uniform data growth.

## 2. Energy diagnostic — `analyze_energies.py`
- xTB **underestimates ΔG by a near-constant +11.5 kcal/mol** (the Δ-learning
  signal), Pearson(xTB,DFT)=0.775, **~7 kcal residual that is flat across structural
  classes** (the irreducible part).
- **Outliers are a functional-group class, not noise:** nitroso σ≈23, nitro σ≈15,
  azo σ≈13 vs the ~7 baseline; hypervalent-S / ketene give unphysical xTB ΔG (to −160).
- orig-500 ≈ new-1000 (no distribution drift from the extrapolation).

## 3. Stricter cleaning v2 — `filter_smiles.py`
Added 3 rejection rules: **isotope**, **zwitterion_or_nitro** (catches nitro), and
**reactive_group** (nitroso/azo/SF5/ketene/isocyanide).
- Pool: 217,975 → **206,966** (`aldehydes_clean_v2.csv`).
- Labeled: 1487 → **1351** (delta_core auto-drops via `filter_smiles.classify`).
- Effect: xgb MAE 3.245 → **3.19** (tail tightened). Cleaning is the highest-value lever.

## 4. THE KEY DISCOVERY — CHO-environment categories
Categorize by what the CHO carbon is bonded to (`cho_category.py`): carbo-aromatic /
hetero-aromatic / vinyl-conjugated / aliphatic — the chemically decisive axis.

**Pool vs labeled mismatch (MaxMin selection bias):**

| class | % of 207k pool | % of labeled | per-category MAE |
|---|---|---|---|
| carbo-aromatic (classic benzoin) | **39%** | **11%** | **2.44** (easiest) |
| aliphatic | 33% | 64% | 3.26 |
| hetero-aromatic | 27% | 21% | 3.34 |
| vinyl/α,β-unsat | 0.4% | 4% | 3.27 (σ high) |

**MaxMin diversity under-samples dense-but-similar classes** (benzaldehydes cluster
tightly in fingerprint space → skipped as "redundant"); aliphatic skeletons get
over-picked. So we labeled the *easiest, most relevant* class least. carbo-aromatic
2.44 is the anomaly (easy); everything else sits at the ~3.3 floor — hetero is not
"broken," carbo is unusually easy.

## 5. Targeted expansion + infra
- **+500 clean** MaxMin expansion: coverage barely moved (nearest-rep p95
  0.745→0.737) → quantity/coverage not the binding constraint.
- **813 carbo-aromatic** (1% of 81,263), **hybrid 406 representative + 407 MaxMin**
  (after the user rightly questioned pure MaxMin; original 200 used PCA+KMeans
  representative sampling — `select_aromatic.py` now does both).
- **250 hetero** (hybrid) — submitted as core (not just diagnostic) after the pivot.
- SLURM moved **rome → genoa** (192 cores/node), array throttle **%192**.

## 6. Model-architecture decisions (all empirically tested)
- **Separate per-category model? NO.** Dedicated carbo-only 2.68 vs global 2.46;
  aromatic-only 3.11 vs global 3.04. The structure-independent correction means
  **pooling all data wins**; splitting fragments data and loses accuracy.
- **Category as a model feature? NO.** ΔMAE ≈ 0 (aromaticity descriptors already
  encode it). Shipped the **per-category applicability-domain table** instead
  (`category_ad.json`: carbo ±2.5 / hetero ±3.3 / aliphatic ±3.3 / vinyl high-var).
- **carbo vs hetero — separate or combined? COMBINED.** carbo 2.68(comb)/2.73(only),
  hetero 3.31(comb)/3.27(only) — differences ±0.04, within noise. One aromatic model.

## 7. THE PIVOT — aromatic-only (user decision)
Restrict the whole project to **aromatic aldehydes (carbo + hetero)**; aliphatic +
vinyl are off-target for benzoin (α-H → enolization/aldol, low yield) — excluded from
**labeling, training, AND prediction**. Cost on aromatic CV: only **+0.07 kcal**.
- `delta_core.CHO_SCOPE = {aromatic_carbo, aromatic_hetero}` (default training filter).
- `src/benzoin_dG/scope.py` + `predict.py`: non-aromatic SMILES short-circuit
  (`benzoin_relevant=False`, no featurize).
- Aliphatic labels **kept on disk but excluded** (not deleted; reversible).
- Future DFT budget → aromatic + hetero only.

**Of the previous 2k (all-category):** aliphatic 1218 / hetero 386 / carbo 218 /
vinyl 75 → only **604 (~31%) aromatic**. ~68% of the budget was off-target. The
aromatic subset is **data-starved** (604) → unlike the global noise floor it's likely
still on the steep part of its learning curve, so the 813+250 batches should
genuinely lower aromatic MAE. Interim aromatic-only baseline: xgb **MAE 3.078** (n=433).

## 8. Process lesson
Cancelled a 79/80-complete sweep on the scope change — **wasteful**: sunk compute
isn't recovered by cancelling. Recovered the best params + baseline (3.159) from
MLflow per-trial logs. Rule: **let long/near-complete jobs finish & supersede**; only
cancel if actively harmful, and ask first. Don't let %192 labeling arrays starve a
co-running training job.

---

## Artifacts created this session
| file | purpose |
|---|---|
| `pipeline/explore_models.py` | regression zoo + uncertainty + classification |
| `pipeline/learning_curve.py` | tree-vs-GNN learning curve |
| `pipeline/analyze_energies.py` | xTB/ORCA energy diagnostics + outliers |
| `pipeline/cho_category.py` | CHO-environment categorization (pipeline source of truth) |
| `pipeline/select_aromatic.py` | category-restricted hybrid selection (carbo/hetero) |
| `pipeline/category_eval.py` | category-feature test + per-category AD |
| `pipeline/slurm/submit_explore.sh` | CPU exploration job (genoa) |
| `src/benzoin_dG/scope.py` + `predict.py` | inference aromatic-scope guard |
| `filter_smiles.py`, `delta_core.py` | v2 cleaning rules + CHO_SCOPE training filter |

Data: `aldehydes_clean_v2.csv`, `pool_categorized.csv`, `subset_v5.csv` (2000),
`subset_aromatic_v1.csv` (813), `subset_hetero_v1.csv` (250).
Figures: `runs/figs/{energy_analysis,model_zoo,learning_curve}.png`.

## In flight / next
- Labeling on genoa: **813 carbo** (`23759042`), **250 hetero** (`23762066`).
- **When both land:** assemble parquet → full **combined aromatic-only sweep**
  (warm-start from recovered params n_est=1200/depth=6/lr=0.023/...) → ship model +
  aromatic AD → confirm the aromatic learning-curve gain → update ARCHITECTURE.md.

## Recovered tuned params (all-category cleaned, cv_mae 3.159) — warm-start seed
`n_estimators=1200, max_depth=6, learning_rate=0.0233, subsample=0.914,
colsample_bytree=0.855, reg_lambda=0.910, reg_alpha=0.0215, min_child_weight=2`
