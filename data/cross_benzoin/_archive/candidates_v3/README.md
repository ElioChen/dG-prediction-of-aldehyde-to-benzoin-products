# Cross-benzoin ΔG candidate dataset v3

## Design goal

This version is designed for computing and later predicting benzoin reaction free energies from aldehyde SMILES. It retains the full structural scope of `aldehydes_clean_v6.csv` instead of pre-excluding aliphatic or flagged aldehydes.

## Files

- `cross_benzoin_aldehydes_v3.csv.gz`: molecule table with all 220,859 source aldehydes and metadata.
- `cross_benzoin_dG_candidates_v3.csv.gz`: 4,000,000 directed reaction candidates from 2,000,000 unique unordered pairs.
- The compressed CSV files are directly readable with pandas (`pd.read_csv(path)`).

## Coverage and sampling

- Classes retained: aliphatic, aromatic_carbo, aromatic_hetero.
- All six unordered class combinations are represented approximately equally within each split.
- Every source aldehyde appears in at least one candidate pair.
- No exclusion based on MW, `xtb_risk`, or aliphatic character.
- Self-pairs are excluded; unordered pairs are unique.

## Directionality and ΔG target

Every unordered pair A+B is expanded into two rows: A as donor/B as acceptor and B as donor/A as acceptor. This is essential because the two orientations can map to different alpha-hydroxy ketone regioisomers and different ΔG values. `dG_kcal_mol` is intentionally blank and `dG_status` is `to_be_computed`.

## Splitting

The split is molecule-disjoint using SHA-256(InChIKey): 80% train, 10% validation, 10% test. Both aldehydes in a reaction belong to the same split, so no source molecule crosses splits.

## Provenance limitation

The input CSV contains only the provenance label `PubChem`. v3 maximizes structural coverage within that input, but it cannot create additional literature/vendor provenance. Adding other databases requires a separate source-enrichment step.

## Reproducibility

Rule version: `cross_benzoin_dG_pairing_v3`; random seed: 20260713.

The reaction and molecule SHA-256 checksums, exact split/class counts and validation
metadata are recorded in `manifest.json`; the rendered checks are in the QA workbook.

## Recommended execution

Do not load or submit all four million rows as one job. Stream this file with
`cross_benzoin/prepare_pair_chunks.py`, enumerate/QC the directed product SMILES with
`cross_benzoin/prepare_product_manifest.py`, then run product QM only for a selected
diversity/uncertainty subset. Reuse the completed aldehyde table with
`cb_featurize.py --aldehyde-cache ... --require-cache-complete`.
