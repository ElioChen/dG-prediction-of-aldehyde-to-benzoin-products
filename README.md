# dG-prediction-of-aldehyde-to-benzoin-products

Predict the Gibbs free energy **ΔG** of the NHC-catalyzed **homo-benzoin
condensation** (`2 R-CHO → R-CH(OH)-C(=O)-R`) directly from an aldehyde
**SMILES**, at DFT accuracy but at a fraction of the DFT cost.

```
ΔG_pred = ΔG_g-xTB + model(QM descriptors, ΔG_g-xTB)     # model ≈ ΔG_DFT(r2SCAN-3c) − ΔG_g-xTB
```

A cheap semi-empirical (g-xTB) estimate of ΔG is corrected up to DFT
(ORCA **r2SCAN-3c**, CPCM/DMSO) quality by a **Δ-learning** ensemble
(MLP + gradient-boosted trees + quantile uncertainty), trained on a
filtered library of ~220k candidate aldehydes and real DFT single-point
labels for nearly the whole library.

## Current best model

**`MORDREDSLIM271_BDEGXTB`** — test MAE **1.503 kcal/mol** (RMSE 2.257, R² 0.875)
on a held-out 70:20:10 split (n≈218k). 275 features: 72 QM/steric descriptors
(xTB + Multiwfn ADCH/QTAIM, reactant + product) + 199 SHAP/correlation-pruned
Mordred descriptors (dispersion/size/shape) + 4 g-xTB bond-dissociation-energy
features. Ships with an uncertainty-routing head: the confident 85% of
predictions hit MAE 1.25; the routed 15% (flagged for DFT follow-up) sit at
MAE ~2.9.

Aromatic substrates predict noticeably better than aliphatic (1.33 vs 1.87
MAE) — historical sampling bias toward aromatics has been confirmed resolved
at full-library scale. The dominant remaining error driver is electronic
(sulfonyl/phosphorus/imine substituents), not geometry or missing descriptors.

## What we tried and why it isn't the production model

- **GNN (graph neural net) on the same 275 features**: a dual-encoder GINE
  architecture (product graph + aldehyde graph, thermodynamic-cycle combine)
  given the exact same feature set as the tabular champion still loses
  standalone (MAE 1.55 vs 1.50) — architecture is not the lever once the
  information content is held fixed. GNN+tabular *stacking* showed a
  promising signal in earlier partial-overlap checks; a full-library,
  leak-free re-verification is in progress before any promotion claim.
- **Pure-SMILES / no-QM-descriptor baselines** (SELFIES bag-of-symbols,
  ECFP fingerprints, a from-scratch sequence model, a from-scratch GINE
  graph model, and a fine-tuned pretrained ChemBERTa): all land in the
  2.7–3.3 kcal/mol MAE range on this task even at full library scale —
  well short of the descriptor-informed Δ-learning model. The QM/xTB layer
  is where most of the achievable accuracy lives, not the architecture
  consuming it.

## Install

```bash
pip install -e .                 # library + inference
pip install -e ".[pipeline]"     # + data-gen / training (MLflow, Optuna, …)
```

External binaries (not pip deps) are discovered at runtime:

```bash
export XTB_BIN=/path/to/xtb                    # required for inference
export MULTIWFN_BIN=/path/to/Multiwfn_noGUI     # optional (ADCH/QTAIM features)
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
src/benzoin_dG/         installable library (import benzoin_dG)
  descriptors.py        1-molecule featurization (wraps _descriptors_backend)
  thermo.py             xTB ΔG (wraps _thermo_backend, ORCA off)
  features.py           assemble + median-impute the model input vector
  predict.py / cli.py   SMILES -> ΔG
  models/                shipped trained model + feature spec
pipeline/                research / data-generation pipeline
  filter_smiles_v6.py    current production SMILES filter (raw -> clean library)
  compute/               xTB/g-xTB/DFT featurization + descriptor workers
  analysis/              model training, finalize_*, GNN experiments, SHAP, diagnostics
  slurm/                 HPC submit scripts for the above
  docs/                  methods notes, weekly reports, data-split docs
cross_benzoin/           cross-substrate (non-homo) benzoin extension
data/library/             filtered aldehyde libraries (v2-v6) + rejects
data/cross_benzoin/       featurization + model outputs for the homo-benzoin study
data/analysis/             plots/reports from various analysis passes
tests/
```

See [FILE_MAP.md](FILE_MAP.md) for a more complete file-by-file index and
[ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline architecture.

## Status (2026-07-13)

Production champion (`MORDREDSLIM271_BDEGXTB`, MAE 1.503) is trained and has
scored the full ~220k-molecule filtered library. GNN and pure-SMILES
alternatives have been explored extensively (see above) and, for this task,
descriptor-informed Δ-learning on gradient-boosted trees remains the
strongest approach. Active work: verifying whether GNN+tabular stacking adds
a real, reproducible gain at full library scale.

**Note on git history:** this repository's local `.git` object database was
found corrupted (silent object-store corruption, not user error) while
preparing this upload; history was restarted from the current, verified-intact
working tree on 2026-07-13 rather than risk pushing corrupted objects.
