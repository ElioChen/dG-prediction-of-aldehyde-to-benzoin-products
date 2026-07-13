# Shipped model artifacts

The trained Δ-learning model is placed here so it travels with the wheel:

- `delta_model.joblib`  — fitted regressor (XGBoost/RF/GBT)
- `feature_list.json`   — exact feature column order
- `metadata.json`       — target, CV metrics, per-feature training medians

Produce them with the pipeline, then copy the winner in:

```bash
python pipeline/sweep_delta.py --trials 60
cp runs/models/{delta_model.joblib,feature_list.json,metadata.json} \
   src/benzoin_dG/models/
```
