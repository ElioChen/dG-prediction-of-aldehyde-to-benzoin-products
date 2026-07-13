# Hybrid D-MPNN (GNN) Δ-learning — experiment log

**Date:** 2026-06-12 · **Data:** `chunk_000` = 500 molecules (204 original + 296 from
the prior MaxMin +300 expansion), **n≈474 after QC**, labels = **r2SCAN-3c** (unified
`featurize.py`, ensemble_k=3) = `featurize_v2.parquet`.

## Label method note (resolved)
The canonical labels are **r2SCAN-3c** (composite, ensemble-averaged) from the unified
featurize path; `chunk_000` ≡ `featurize_v2.parquet` (474/474). An older PBE0-D4 set
(`data/raw/labels`, `ml/labels_expand300`) also exists and differs by ~2.6 kcal median
— a **method difference, not corruption**. (A mid-session detour that rebuilt chunk_000
to PBE0-D4 was reverted; numbers below are r2SCAN-3c. The PBE0-D4 run gave the same
verdict — trees ~3.2 / GNN ~3.5 — so the conclusion is method-robust.)

## Question
Does a 2D message-passing GNN (Chemprop v2 D-MPNN), with the graph **plus** the 62
descriptors + `dG_xtb` as extra features (`x_d`), beat the tree Δ-learning model —
and does the A100 budget pay off? (Revisiting the deferral in
[delta_vs_direct.md](delta_vs_direct.md).)

## Setup
- Same target (`dG_orca − dG_xtb`), same `delta_core` table, **same RepeatedKFold(seed=42)
  folds** as the trees → directly comparable OOF metrics.
- Hybrid: aldehyde SMILES → BondMessagePassing → MeanAggregation, concat `x_d`
  (StandardScaler, leakage-safe per fold) → RegressionFFN. Target also standardized
  per fold. Code: [pipeline/gnn/](../gnn/). Env: `envs/gnn` (chemprop 2.2.3, torch 2.12+cu130).

## Result — GNN does NOT beat trees at n≈474 (r2SCAN-3c)

| model | CV MAE | notes |
|---|---|---|
| xTB baseline (correction ≡ 0) | **11.9** | target to beat |
| **trees** xgb / rf / gbt (3×5 CV) | **3.11 / 3.14 / 3.15** | R² ≈ 0.65 |
| hybrid D-MPNN (9-trial Optuna, 2×5 CV) | 3.34 best | ~0.2–0.3 worse than trees |
| (PBE0-D4 detour, for the record) | trees 3.20 / GNN 3.51 | same verdict, older method |

The corrupt-label sweep already showed the D-MPNN robust at 3.3–3.6 across very
different hyperparameters → a **data-size ceiling, not a tuning problem**, consistent
with the prior prediction that NNs need more data than trees. The corrected-label
single train confirms the level (see table). (GPU budget is tight; sweep was capped at
9/40 trials, ~2.5 min/trial on one A100.)

## Learning curve (the real test of the GNN+A100 bet)
As the r2SCAN-3c set grows (unified featurize, `data/featurize.parquet`):

| n (post-QC) | tree xgb MAE / R² | hybrid D-MPNN MAE / R² | gap |
|---|---|---|---|
| 474  | 3.11 / 0.66 | 3.34 / 0.59 (sweep-best) | +0.23 |
| 1038 | 3.27 / 0.66 | 3.49 / 0.63 (ens=3)       | +0.22 |
| 1427 | 3.21 / 0.67 | 3.40 / 0.63 (ens=3)       | +0.19 |

**The GNN tracks the trees at a ~constant ~0.2 kcal/mol deficit; 3× data did not
close it.** No GNN crossover. See `runs/figs/learning_curve.png`.

**Model-zoo verdict (n=1427, `explore_models.py`):** 15 regression families all land
in MAE 3.23–3.44 — and plain **ridge ties the best (3.23)**, with xgb/rf/gpr/stack
all ≈3.24. No family beats the trees; even a linear model reaches the floor. The
energy diagnostic explains it: the xTB→DFT correction is a near-constant +11.5
offset with ~7 kcal structure-independent residual noise. **~3.2 kcal MAE is a
noise floor, not a model-capacity limit** → the lever is label quality (the v2
exotic-group cleaning) and conformer/geometry noise, NOT model choice or more data.
For active learning, **GPR gives the best-calibrated σ** (PICP80 0.83, NLL 2.88 vs
ngboost 0.69 / quantile-GBM 0.54). Favourability classifier (ΔG<0): AUC 0.846.

## Recommendation
1. **Trees remain the production model at this data size** (n≈464). Keep shipping the
   tree Δ-model; rebuild its labels with `merge_labels.py` (NOT the buggy 17:07 path).
2. **The GNN + A100 bet rests entirely on the learning curve.** Its ~0.1–0.3 kcal deficit
   is the quantity expected to flip as n grows. Expansion to ~1000 (selected:
   `subset_expansion_v2.csv`, +500) → label → re-plot **GNN-vs-tree learning curve**.
   Only a clear crossover justifies continuing toward several thousand.

## Reproduce
```bash
sbatch pipeline/slurm/submit_gnn.sh                                    # 1 train (cheap)
sbatch --export=ALL,MODE=sweep,TRIALS=40 pipeline/slurm/submit_gnn.sh  # full sweep (~2h, opt-in)
```
