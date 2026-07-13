# Hybrid D-MPNN Δ-learning (pipeline/gnn/)

A 2D message-passing GNN (Chemprop v2) alternative to the tree Δ-learning model.
The aldehyde **graph** is the message-passing input; the **62 descriptors + dG_xtb**
(exactly `delta_core`'s feature matrix `X`) ride along as Chemprop *extra datapoint
descriptors* `x_d`, concatenated into the FFN readout. Target = the DFT correction
`dG_orca − dG_xtb`; prediction = `dG_xtb + model(graph, x_d)`.

It reuses `delta_core.load_training_table` (same QC, same `X`, same target) and the
**same** `RepeatedKFold(seed)` splits as the trees, so out-of-fold predictions are
directly comparable to `train_delta.py` / `sweep_delta.py` and land in the **same**
MLflow experiment (`benzoin_delta_dG`).

## Env (one-time, dedicated — heavy CUDA torch)

Built with micromamba at `/gpfs/scratch1/shared/schen3/envs/gnn`
(python 3.12, chemprop 2.2.3, torch 2.12+cu130). Equivalent to `pip install -e ".[gnn]"`
into a fresh py3.12 env. The SLURM script points `$ENV` at it.

## Run

```bash
PY=/gpfs/scratch1/shared/schen3/envs/gnn/bin/python
export PYTHONPATH=$PWD/..            # so delta_core imports (or run via submit_gnn.sh)

# one config, 3x5 CV, logged to MLflow:
$PY train_gnn.py --params '{"ensemble":3,"depth":4}'

# Optuna sweep (run on the A100):
sbatch ../slurm/submit_gnn.sh                       # MODE=sweep, 40 trials
MODE=train PARAMS='{"depth":5,"ensemble":3}' sbatch ../slurm/submit_gnn.sh
```

GPU budget is tight: the GNN trains in minutes, so the A100 goes to breadth
(trials × ensemble), not long single trains. Keep `--time` modest.

## Files
- `gnn_core.py` — `delta_core.TrainTable` → Chemprop bridge; leakage-safe per-fold
  StandardScaler on `x_d` and `y`; `cv_evaluate_gnn` (fold-matched to the trees);
  `fit_full` for the final shipped model.
- `train_gnn.py` — one config, CV, parity figure, MLflow logging, save model+scalers.
- `sweep_gnn.py` — Optuna search (depth/width/dropout/ensemble/…), nested MLflow runs.
- `../slurm/submit_gnn.sh` — gpu_a100 launcher (MODE=sweep|train).
