# benzoin-dg status

_Last updated: 2026-07-20._

This is the repository's authoritative status page. README is the short
user-facing story; this file records what is shipped, what is experimental, and
what must be verified before promotion.

## Production paths

| Path | Artifact | Status | Notes |
|---|---|---|---|
| `benzoin-dg` / `predict_dG()` | `src/benzoin_dG/models/delta_model.joblib` | default package path | Older 63-feature xGB model; `metadata.json` reports n=1644, 5x3 CV MAE 2.000, g-xTB baseline. |
| `benzoin-dg --champion` / `predict_dG_champion()` | `pipeline/models/gxtb_dft_correction_ENSEMBLE72_20260626.joblib` | current full-library champion path | 72-feature ensemble inference adapter; reported test MAE 1.503 kcal/mol on the 219k-label homo campaign. |
| `benzoin-dg --fast` / `predict_dG_fast()` | `src/benzoin_dG/models/surrogate_model.joblib` | screening tier | Pure 2D/RDKit surrogate. Use for pre-screening, not final ranking. |

The package scope is aromatic homo-benzoin aldehydes. The default and fast paths
reject aliphatic and alpha,beta-unsaturated aldehydes with
`benzoin_relevant=False`. The champion path uses the modern `cross_benzoin`
funnel and should be treated as the accuracy path when its external runtime is
available.

## Runtime dependencies

- Default delta path: xTB plus the local g-xTB bridge when the model baseline is
  `gxtb_cosmo_dmso`; Multiwfn is optional and missing ADCH/QTAIM-like features
  are median-imputed.
- Champion path: xTB, the local `cross_benzoin`/`pipeline/compute` code, the
  champion joblib bundle in `pipeline/models/`, and
  `../envs/bde_lite/bin/python` for numpy-2-compatible unpickling.
- Fast path: RDKit/model artifacts only; no xTB or Multiwfn.

## Confirmed research result, not packaged

- GNN + tabular stacking is the current best research estimate: fixed 40% GNN +
  60% tabular blend, reported test MAE 1.427 kcal/mol on a leak-free held-out
  split. It is not wired into the installable package because the GNN checkpoint
  and dual-input prediction path are not bundled.

## Candidate / active research

- `MORDREDSLIM271_BDEGXTB`: promising 275-feature tabular research result. It
  should not replace the champion CLI path until a matching inference adapter,
  feature spec, metadata record, and smoke test are shipped together.
- Uncertainty routing: the reported confident/routed MAE split should be
  regenerated from frozen artifacts and recorded here.

## Retired / legacy

- PBE0-D4 labels and pre-unified descriptor/label paths are incompatible with
  the r2SCAN-3c production line and must not be mixed into current training.
- Older June 2026 architecture notes are retained for provenance; production
  model statements in those sections are superseded by this file.

## Next steps

1. Add model-card metadata for each promotable artifact: input table path/hash,
   split seed, label level, baseline, feature list hash, training command,
   commit SHA, and metrics.
2. Re-run local smoke tests on `/scratch-shared/schen3/benzoin-dg`: `pytest`,
   `benzoin-dg "O=Cc1ccccc1" --fast --json`, and one standard xTB/champion
   prediction where external binaries are installed.
3. Quantify Multiwfn-missing behavior on a fixed validation subset.
4. Finish uncertainty-routing calibration for the shipped champion path.
5. Package stacking only after adding the GNN checkpoint, dual-input inference,
   metadata, and smoke tests for single-molecule use.

## Local audit notes

On 2026-07-20 the local working copy had only untracked experimental/data files
before these cleanup changes. Existing untracked files were left untouched.
