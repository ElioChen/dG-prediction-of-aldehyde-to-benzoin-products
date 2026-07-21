# cross_benzoin/ — architecture & naming conventions

Clean, single-entry pipeline for NHC benzoin **product** featurization (homo =
diagonal special case of cross). Replaces the scattered
`pipeline/compute/{featurize_screen,featurize_product,backfill_multiwfn,merge_multiwfn}.py`
for everything downstream of the aldehyde library. Reuses the validated QM
backends (does NOT re-implement them):

- `pipeline/compute/ald_descriptors_qm.py` — xTB / morfeus / Multiwfn calculators
- `pipeline/compute/thermo_orca.py`        — conformer ranking, xTB `--ohess` G
- `pipeline/compute/conf_funnel_v3.py`     — funnel_v3 conformer search (topology guard)

## Principle

ONE method, everything saved. Aldehyde and product are featurized with the SAME
conformer method (funnel_v3) and identical descriptor backends, and all geometries
+ energies + descriptors are persisted and cross-linked by a stable ID.

## Stable IDs  (the linking key — never an enumerate index)

- `ald_id`  : stable per-aldehyde id taken from the library column `index`
              (falls back to InChIKey). Used in aldehydes.csv and xyz filenames.
- `pair_id` : `"{donor_id}__{acceptor_id}"` (ordered). HOMO (donor==acceptor)
              collapses to the single `donor_id` (no redundant `<id>__<id>`).
              Used in products.csv and product xyz filenames.

## File / directory layout

```
benzoin-dg/cross_benzoin/                # code (this package)
  ARCHITECTURE.md
  cb_schema.py          # column lists + naming helpers (single source of truth)
  cb_featurize.py       # the one entry point (aldehyde and/or product)
  slurm/submit_cb_featurize_array.sh
  analysis/

benzoin-dg/data/cross_benzoin/<run>/     # outputs (e.g. <run> = homo_v6, cross_core_v1)
  aldehydes.csv         # one row per aldehyde   (key: ald_id)
  products.csv          # one row per pair        (key: pair_id, + donor_id, acceptor_id)
  xyz_ald/ald_<ald_id>.xyz
  xyz_prod/prod_<pair_id>.xyz   (homo: prod_<id>.xyz; cross: prod_<donor>__<acceptor>.xyz)
  chunk_*/              # (array mode) per-task aldehydes.csv/products.csv/xyz_*; merged after
```

## File naming (unified)

| artifact | name |
|---|---|
| aldehyde geometry | `xyz_ald/ald_<ald_id>.xyz` |
| product geometry  | `xyz_prod/prod_<pair_id>.xyz   (homo: prod_<id>.xyz; cross: prod_<donor>__<acceptor>.xyz)` |
| aldehyde table    | `aldehydes.csv` |
| product table     | `products.csv` |
| concatenated (array) | `aldehydes_all.csv`, `products_all.csv` |

## Column conventions

Shared metadata (identical names in both tables):
`id` (ald_id or pair_id), `smiles`, `xtb_optimized`, `error`, `xyz_file`, `G_xtb` (Eh).

Aldehyde descriptors — single carbonyl site, suffix `_CHO_C` / `_CHO_O`
(same as the screen schema, so existing analysis still works).

Product descriptors — two derived sites + the H-bond, suffixes:
`_ketC` `_ketO` (former donor carbonyl → ketone), `_carbC` `_hydO` `_hydH`
(former acceptor carbonyl → carbinol), bonds `_CO_ket` `_CC_new` `_CO_carb`,
H-bond `hb_dist` `hb_angle` `qtaim_rho_HB`. ΔG column `dG_xtb_kcal`.

## Data flow

```
aldehyde library CSV (index, SMILES, ...)
        │   pairs CSV: donor_id, acceptor_id, donor_smiles, acceptor_smiles
        ▼
prepare_product_manifest.py  (directed product generation + sanitation/QC)
        ▼ valid products only
cb_featurize.py --aldehyde-cache existing_aldehydes.csv --require-cache-complete
        ├─ existing aldehydes  → reuse GFN2/g-xTB free energies and descriptors
        └─ per directed pair   → products.csv row + xyz_prod/prod_<did>__<aid>.xyz
                                 ΔG = G(prod) − G(donor) − G(acceptor)
```

## Migration / status

- v6 homo campaign and all future cross runs use THIS package.
- Old scripts kept for provenance; `backfill_multiwfn`/`merge_multiwfn` are
  deprecated (plain-geometry, method-inconsistent).
- See [[cross-benzoin-pipeline-handoff]], [[multiwfn-env-and-screen-gap]].

## Directed candidate library v3

`data/cross_benzoin/candidates_v3/` contains two million unique unordered pairs expanded
to four million donor/acceptor orientations, with complete coverage of the 220,859-source
aldehyde library. The compressed source manifest is intentionally unlabeled.

Use `prepare_pair_chunks.py` to stream the `.csv.gz` into bounded, uncompressed manifests
with the `donor_id,acceptor_id,donor_smiles,acceptor_smiles` schema consumed by
`cb_featurize.py --pairs`. Generated chunks are execution artifacts and are not tracked.

Run `prepare_product_manifest.py` on each chunk before QM. This explicitly validates the
two directed regioisomers and prevents invalid products from entering the expensive
conformer/frequency workflow.

Modeling and validation recommendations for transferring the homo model to this directed
cross space are documented in `docs/CROSS_BENZOIN_ML_RECOMMENDATIONS.md`; implementation
gates and descriptor roles are in `docs/NEXT_STEPS.md` and
`docs/DESCRIPTOR_POLICY_CROSS.md`.
