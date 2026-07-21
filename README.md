# Benzoin ΔG prediction from aldehyde SMILES

This repository is evolving into a unified predictor for directed **homo- and
cross-benzoin** reaction free energies. The intended input is `(donor SMILES,
acceptor SMILES)`; homo-benzoin is the diagonal special case `(A, A)`. The shipped
inference model is currently homo-only. The cross extension now includes a 4M-row
candidate release, deterministic product enumeration/QC, reusable quantum-computation
workflows and a role-aware modeling plan.

```
ΔG_pred = ΔG_g-xTB + model(QM descriptors, ΔG_g-xTB)     # model ≈ ΔG_DFT(r2SCAN-3c) − ΔG_g-xTB
```

A cheap semi-empirical (g-xTB) estimate of ΔG is corrected up to DFT
(ORCA **r2SCAN-3c**, CPCM/DMSO) quality by a **Δ-learning** ensemble
(MLP + gradient-boosted trees + quantile uncertainty), trained on a
filtered library of ~220k candidate aldehydes and real DFT single-point
labels for nearly the whole library.

## Current best model

**GNN (40%) + tabular (60%) stacking** — CONFIRMED test MAE **1.427 kcal/mol**
(vs tabular-only 1.485) on a held-out, leak-free test split (job 24578348,
2026-07-13). The blend combines the tabular champion below with a dual-encoder
GINE GNN (product graph + aldehyde graph, thermodynamic-cycle combine) via a
fixed `w_gnn=0.40` weight fit on out-of-fold predictions. Full-library scoring
(`data/cross_benzoin/homo_v6/viz_gxtb_20260625/products_dG_corrected_GNNSTACK_w40_20260714.csv`,
n=218,227, GNN cache coverage 99.9%) is the recommended best-estimate column
(`dG_blend_final`) going forward. **This blend is a research-pipeline result,
not yet wired into the installable `benzoin-dg` package** — `predict_dG()`
still returns the tabular-only model below (see Install/Use and Status);
shipping the blend requires bundling the GNN checkpoint and a dual-input
predict path, which is open work.

Underlying tabular model, **`MORDREDSLIM271_BDEGXTB`** — test MAE **1.503
kcal/mol** (RMSE 2.257, R² 0.875) on a held-out 70:20:10 split (n≈218k). 275
features: 72 QM/steric descriptors (xTB + Multiwfn ADCH/QTAIM, reactant +
product) + 199 SHAP/correlation-pruned Mordred descriptors (dispersion/size/
shape) + 4 g-xTB bond-dissociation-energy features. Ships with an
uncertainty-routing head: the confident 85% of predictions hit MAE 1.25; the
routed 15% (flagged for DFT follow-up) sit at MAE ~2.9. This is available through the
research/pipeline artifact path and the `--champion` adapter; the default
`src/benzoin_dG/models/` artifact is older and kept for compatibility.

Aromatic substrates predict noticeably better than aliphatic (1.33 vs 1.87
MAE) — historical sampling bias toward aromatics has been confirmed resolved
at full-library scale. The dominant remaining error driver is electronic
(sulfonyl/phosphorus/imine substituents), not geometry or missing descriptors.

## What we tried and why it isn't the production model

- **GNN (graph neural net) on the same 275 features**: a dual-encoder GINE
  architecture (product graph + aldehyde graph, thermodynamic-cycle combine)
  given the exact same feature set as the tabular champion still loses
  standalone (MAE 1.55 vs 1.50) — architecture is not the lever once the
  information content is held fixed. GNN+tabular *stacking*, however, is a
  confirmed, reproducible gain at full-library scale (1.503 → 1.427; see
  "Current best model" above) — complementary errors between the two model
  families are real, not an artifact of the earlier partial-overlap checks.
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
benzoin-dg "O=Cc1ccccc1" --champion      # full-library champion path
benzoin-dg "O=Cc1ccccc1" "O=CCC" --json
```

The second command evaluates two independent **homo** inputs; it is not yet a directed
cross pair. The planned pair API is
`predict_cross_dG(donor_smiles, acceptor_smiles)` after cross-label/model validation.

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
[ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline architecture. See
[STATUS.md](STATUS.md) for the authoritative production/candidate/legacy split.

## Status (2026-07-14)

The homo GNN+tabular stacking blend (MAE 1.427) is confirmed at full-library
scale and is the current best *research* estimate; the shipped
`benzoin-dg --champion` exposes the tabular-only champion
(`MORDREDSLIM271_BDEGXTB`, MAE 1.503), while default `benzoin-dg` remains the
older compatibility artifact. The blend is not packaged for single-molecule
inference yet. Pure-SMILES alternatives (SELFIES, ECFP,
from-scratch GINE, fine-tuned ChemBERTa) have all been benchmarked at full
library scale and land at 2.7–3.3 MAE — well short of descriptor-informed
Δ-learning, confirming the QM/xTB descriptor layer (not model architecture)
is where the achievable accuracy lives. Open work: (1) package the GNN+tabular
blend behind `predict_dG()`, (2) compute real labels for a cross-benzoin
(donor ≠ acceptor) sample — the 4M-row v3 candidate set below is still
entirely unlabeled, so cross-substrate accuracy is currently unmeasured, and
(3) role-aware (donor/acceptor) descriptors for the cross extension are
specified in `NEXT_STEPS.md` but not yet computed.

## Cross-benzoin extension

The versioned candidate release under
[`data/cross_benzoin/candidates_v3/`](data/cross_benzoin/candidates_v3/README.md)
contains all 220,859 source aldehydes and **four million directed** cross-benzoin
candidates (two million unique unordered pairs). Aliphatic aldehydes are retained. The
release is unlabeled and intended for the
existing GFN2/g-xTB/DFT computation workflow. Use
[`cross_benzoin/prepare_pair_chunks.py`](cross_benzoin/prepare_pair_chunks.py) to create
bounded manifests, then
[`cross_benzoin/prepare_product_manifest.py`](cross_benzoin/prepare_product_manifest.py)
to enumerate and validate the directed products before expensive computation.

Model-transfer, fusion-validation and evaluation recommendations are collected in
[`CROSS_BENZOIN_ML_RECOMMENDATIONS.md`](cross_benzoin/docs/CROSS_BENZOIN_ML_RECOMMENDATIONS.md).
The executable plan and descriptor decisions are in
[`NEXT_STEPS.md`](cross_benzoin/docs/NEXT_STEPS.md) and
[`DESCRIPTOR_POLICY_CROSS.md`](cross_benzoin/docs/DESCRIPTOR_POLICY_CROSS.md).

**Note on git history:** this repository's local `.git` object database was
found corrupted (silent object-store corruption, not user error) while
preparing this upload; history was restarted from the current, verified-intact
working tree on 2026-07-13 rather than risk pushing corrupted objects.
