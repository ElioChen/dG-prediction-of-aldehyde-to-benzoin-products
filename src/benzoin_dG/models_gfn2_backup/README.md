# Shipped model artifacts

Two prediction tiers travel with the wheel.

**Accurate tier — Δ-learning (needs xTB), `predict_dG`:**

- `delta_model.joblib`  — fitted regressor (XGBoost/RF/GBT)
- `feature_list.json`   — exact feature column order
- `metadata.json`       — target, CV metrics, per-feature training medians
- `ad_reference.npz`    — applicability-domain reference (NN distances)
- `category_ad.json`    — per-category (carbo/hetero) OOF MAE table

Produce them with the pipeline, then copy the winner in (`assemble_model.py` does
the copy + builds `ad_reference.npz`):

```bash
python pipeline/sweep_delta.py --trials 60      # or: train_delta.py --model xgb
python pipeline/assemble_model.py
```

**Fast tier — pure-SMILES 2D surrogate (no xTB), `predict_dG_fast`:**

- `surrogate_model.joblib`    — fitted regressor on 26 geometry-free RDKit-2D descriptors
- `surrogate_features.json`   — feature column order
- `surrogate_metadata.json`   — CV metrics + per-feature medians

Predicts r2SCAN-3c ΔG directly from the SMILES (no conformer search, no quantum) at
~1 kcal higher MAE than the Δ-model — a pre-filter for large candidate sets. Ship with:

```bash
python pipeline/train_surrogate.py --save
```
