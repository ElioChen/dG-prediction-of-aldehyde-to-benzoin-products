# MLflow experiment tracking — benzoin Δ-learning

All training runs (stage D) log to a **local SQLite store** — no server needed.
(MLflow 3.x put the plain file store in maintenance mode and errors on it, so we
use SQLite for the metadata + a local folder for artifacts.)

- Tracking URI:    `sqlite:///scratch-shared/schen3/ml/mlflow.db`
- Artifact root:   `file:/scratch-shared/schen3/ml/mlartifacts`
- Experiment:      `benzoin_delta_dG`

Both scripts call `delta_core.setup_mlflow()`, which wires this up and creates the
experiment on first use — you don't set any of it manually.

## Scripts

| Script | What it does |
|--------|--------------|
| `delta_core.py`   | Shared data loading + repeated-K-fold CV (no side effects). |
| `train_delta.py`  | Train **one** config → 1 MLflow run (params, CV metrics, parity + SHAP figs, model). |
| `sweep_delta.py`  | **Optuna** search over models/hyperparams → 1 nested run per trial + a parent run holding the best model. |

```bash
source /home/schen3/venv/nhc-workflow/bin/activate
cd /scratch-shared/schen3

# single baseline run
python ml/train_delta.py                       # xgb defaults
python ml/train_delta.py --model rf --repeats 5

# hyperparameter / model-family search (recommended for "fit it well")
python ml/sweep_delta.py --trials 60                       # searches xgb+rf+gbt
python ml/sweep_delta.py --trials 80 --model xgb           # pin to xgb
python ml/sweep_delta.py --trials 60 --target dG_orca_shermo_kcal
```

Objective = **repeated 5×3-fold CV MAE** of the Δ-learning ΔG vs the DFT target.
Repeated K-fold is deliberate: at n≈200 a single 5-fold MAE is too noisy to rank
configs, so each score averages out-of-fold predictions over 3 shuffles.

## Key metrics logged per run

- `cv_mae` / `cv_rmse` / `cv_r2` — Δ-learning vs DFT ΔG (the numbers that matter)
- `base_mae` / `base_rmse` / `base_r2` — pure-xTB baseline (correction ≡ 0)
- `mae_improvement` = `base_mae − cv_mae` (how much DFT-level accuracy Δ-learning buys)

A run is "good" only if `cv_mae` is clearly below `base_mae`; otherwise the
descriptors aren't explaining the xTB→DFT gap and stages E/F shouldn't proceed.

## Viewing the UI (over SSH)

The cluster has no browser, so port-forward the MLflow UI to your laptop:

```bash
# on your LAPTOP — opens an SSH tunnel and forwards port 5000
ssh -L 5000:localhost:5000 schen3@<cluster-login-host>

# then, in that SSH session, on the cluster:
source /home/schen3/venv/nhc-workflow/bin/activate
mlflow ui --backend-store-uri sqlite:///scratch-shared/schen3/ml/mlflow.db --port 5000
```

Open <http://localhost:5000> on your laptop → experiment `benzoin_delta_dG`.
Sort by `cv_mae`, select runs → **Compare** for parallel-coordinate / contour plots.

Quick CLI peek without the UI:

```bash
mlflow runs list --experiment-name benzoin_delta_dG \
  --tracking-uri sqlite:///scratch-shared/schen3/ml/mlflow.db
```
