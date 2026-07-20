# benzoin-dg implementation report

_Generated: 2026-07-20_

This report summarizes the files created or modified in this Codex session for
the benzoin-dg status/packaging cleanup.

## Scope

The implementation focused on making the repository state easier to interpret:
separating the default compatibility model, the tabular champion path, the fast
screening tier, and the confirmed but unpackaged GNN+tabular research result.

Existing untracked experimental/data files were left untouched.

## New files

### `STATUS.md`

Purpose: authoritative project status page.

Content added:

- Production path table:
  - default `benzoin-dg` / `predict_dG()` uses `src/benzoin_dG/models/delta_model.joblib`.
  - `benzoin-dg --champion` / `predict_dG_champion()` uses `pipeline/models/gxtb_dft_correction_ENSEMBLE72_20260626.joblib`.
  - `benzoin-dg --fast` / `predict_dG_fast()` uses the 2D screening surrogate.
- Runtime dependency notes for default, champion, and fast paths.
- Confirmed research result section for GNN+tabular stacking, reported MAE 1.427, not yet packaged.
- Candidate/active research notes for `MORDREDSLIM271_BDEGXTB` and uncertainty routing.
- Legacy warning that PBE0-D4 and pre-unified descriptor/label paths must not be mixed into the r2SCAN-3c line.
- Next-step checklist for model cards, tests, Multiwfn-missing quantification, uncertainty calibration, and packaging stacking.

### `IMPLEMENTATION_REPORT_20260720.md`

Purpose: this generated summary of all new files and modified content.

## Modified files

### `README.md`

Changes:

- Clarified that the best current research estimate is still the GNN+tabular stacking blend, but it is not packaged for single-molecule inference.
- Corrected the tabular champion packaging description:
  - `benzoin-dg --champion` exposes the tabular-only champion adapter.
  - default `benzoin-dg` remains the older compatibility artifact.
- Added a `--champion` command example.
- Added a link to `STATUS.md` as the authoritative production/candidate/legacy split.

### `ARCHITECTURE.md`

Changes:

- Added a top-level status note saying the file preserves historical pipeline architecture.
- Pointed readers to `STATUS.md` for current production/candidate/legacy status.
- Marked older production-model statements in the architecture document as superseded by `STATUS.md`.

### `src/benzoin_dG/cli.py`

Changes:

- Imported `predict_dG_champion`.
- Added `_as_dict()` so JSON output can serialize both `Prediction` and `ChampionPrediction` dataclasses.
- Added `_format_champion()` for non-JSON champion output.
- Added CLI options:
  - `--champion`
  - `--xtb-cores`
- Added validation that `--fast` and `--champion` cannot be used together.
- Routed CLI calls:
  - `--champion` -> `predict_dG_champion(...)`
  - `--fast` -> `predict_dG_fast(...)`
  - default -> `predict_dG(...)`

### `src/benzoin_dG/predict.py`

Changes:

- Extended `Prediction` with:
  - `model_version`
  - `scope_note`
- `_load_model()` now derives a compact model version string from `metadata.json`, including model family, sample count, feature count, and baseline.
- Default prediction errors now include more actionable runtime guidance for:
  - missing g-xTB/pipeline bridge
  - failed g-xTB ΔG
  - failed xTB ΔG
- Successful default and fast predictions now include `model_version`.
- Human-readable formatting now prints the model version when present.
- Fast-path text now explicitly marks the 2D surrogate as a screening tier.

### `tests/test_smoke.py`

Added smoke/contract tests:

- `test_package_model_metadata_matches_feature_list`
  - checks `metadata.json["n_features"] == len(feature_list.json)`.
  - checks the shipped default baseline is `gxtb_cosmo_dmso`.
- `test_out_of_scope_short_circuits_without_model`
  - verifies aliphatic aldehyde input short-circuits before model loading.
- `test_cli_rejects_conflicting_tiers`
  - verifies `--fast --champion` returns exit code 2 and a clear error.

## Verification

Commands run:

```bash
/scratch-shared/schen3/envs/bde_gnn/bin/python -m py_compile \
  src/benzoin_dG/cli.py src/benzoin_dG/predict.py
```

Result: passed.

```bash
/scratch-shared/schen3/envs/bde_gnn/bin/python - <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from benzoin_dG.cli import main
models = Path('src/benzoin_dG/models')
feats = json.loads((models / 'feature_list.json').read_text())
meta = json.loads((models / 'metadata.json').read_text())
assert meta['n_features'] == len(feats), (meta['n_features'], len(feats))
assert meta['baseline'] == 'gxtb_cosmo_dmso'
assert main(['O=Cc1ccccc1', '--fast', '--champion']) == 2
print('light checks ok')
PY
```

Result: passed.

Full `pytest` was not run because:

- `pytest` is not installed on the system Python.
- `/scratch-shared/schen3/envs/gnn/bin/python` fails during interpreter startup.
- Other available environments either lack `pytest` or are not compatible with the package imports.

## Git status after changes

Tracked files modified:

- `ARCHITECTURE.md`
- `README.md`
- `src/benzoin_dG/cli.py`
- `src/benzoin_dG/predict.py`
- `tests/test_smoke.py`

New files created by this session:

- `STATUS.md`
- `IMPLEMENTATION_REPORT_20260720.md`

Pre-existing untracked files left untouched:

- `cross_benzoin/relabel_scaffold_split_8rounds.py`
- `cross_benzoin/slurm/submit_assemble_training_table_v3.sh`
- `cross_benzoin/slurm/submit_relabel_8rounds.sh`
- `data/cross_benzoin/cross_round7/gnn_arch_sweep/...`
- `data/cross_benzoin/cross_round8/cross_round8_dft_products.csv`
- `data/cross_benzoin/cross_round8/round8_products_mordred.csv`
- `pipeline/slurm/test_write.tmp`
- `thought.md`

## Remaining work

- Install or repair a test environment with `pytest`, RDKit, joblib, xgboost, and the package dependencies.
- Run full `pytest`.
- Run `benzoin-dg "O=Cc1ccccc1" --fast --json`.
- Run one real `benzoin-dg "O=Cc1ccccc1" --champion` smoke test on a node with the required xTB and bde_lite runtime available.
- Add full model-card metadata for champion, default, surrogate, and future stacking artifacts.
