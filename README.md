# benzoin_dG

Predict the Gibbs free energy **ΔG** of the **benzoin condensation**
(`2 R-CHO → R-CH(OH)-C(=O)-R`) directly from an aldehyde **SMILES**.

A cheap xTB ΔG is corrected up to DFT (ORCA **r2scan-3c**, CPCM/DMSO)
quality by a **Δ-learning** model:

```Shell
ΔG_pred = ΔG_xTB + model(descriptors, ΔG_xTB)     # model ≈ ΔG_DFT − ΔG_xTB
```

## Install

```Shell
pip install -e .                 # library + inference
pip install -e ".[pipeline]"     # + data-gen / training (MLflow, Optuna, …)
```

External binaries (not pip deps) are discovered at runtime — set if not on PATH:

```bash
export XTB_BIN=/path/to/xtb              # required for inference
export MULTIWFN_BIN=/path/to/Multiwfn_noGUI   # optional (ADCH/QTAIM features)
```

Without Multiwfn the ADCH/QTAIM descriptors fall back to training medians.

## Use

```bash
benzoin-dg "O=Cc1ccccc1"                 # benzaldehyde
benzoin-dg "O=Cc1ccccc1" "O=CCC" --json
```

```python
from benzoin_dG import predict_dG
p = predict_dG("O=Cc1ccccc1")
print(p.dG_pred, p.dG_xtb, p.dG_correction)
```

## Layout

```
src/benzoin_dG/        installable library (import benzoin_dG)
  descriptors.py       1-molecule featurization  (wraps _descriptors_backend)
  thermo.py            xTB ΔG  (wraps _thermo_backend, ORCA off)
  features.py          assemble + median-impute the model input vector
  predict.py / cli.py  SMILES -> ΔG
  models/              SHIPPED trained model + feature spec
pipeline/              research / data-generation (not shipped with the wheel)
  filter_smiles.py     deep SMILES filter of the raw library
  select_subset.py     fingerprint → PCA → KMeans representative subset
  train_delta.py       train ONE config, log to MLflow
  sweep_delta.py       Optuna search over models/hyperparams (MLflow)
  delta_core.py        shared data-load + repeated-K-fold CV
  slurm/  docs/        HPC submit scripts; filter & MLflow notes
data/                  generated inputs/outputs (gitignored)
tests/
```

The vendored `_descriptors_backend.py` / `_thermo_backend.py` are the exact
scripts used to build the training set, so inference featurization matches
training by construction.

## Build the model from scratch

```bash
# 1 filter raw library → clean aldehydes
python pipeline/filter_smiles.py --input <raw.csv> --output data/library/aldehydes_clean.csv
# 2 pick a representative subset
python pipeline/select_subset.py --input data/library/aldehydes_clean.csv --out data/library/subset.csv
# 3,4 descriptors + ΔG labels on the subset  (HPC)
sbatch pipeline/slurm/submit_descriptors.sh
sbatch pipeline/slurm/submit_labels.sh
# 5 search + train  (writes runs/models/delta_model.joblib)
python pipeline/sweep_delta.py --trials 60
# 6 ship the winner
cp runs/models/{delta_model.joblib,feature_list.json,metadata.json} src/benzoin_dG/models/
```

See [pipeline/docs/MLFLOW.md](pipeline/docs/MLFLOW.md) for experiment tracking
and [pipeline/docs/benzoin_filters.md](pipeline/docs/benzoin_filters.md) for the
library-filtering rules.

## Status

Pipeline + package scaffold complete. The shipped model is pending the ΔG
label job (stage C) and the Optuna training run (stage D).