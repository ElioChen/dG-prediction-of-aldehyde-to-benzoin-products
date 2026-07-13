# Cross-benzoin directed candidate set v2

This directory contains an unlabeled, computation-ready candidate set for extending the
homo-benzoin workflow to cross-benzoin reactions. Homo-benzoin is the diagonal case
`donor == acceptor`; this dataset covers the off-diagonal pair space as well.

## Contents

| File | Contents |
|---|---|
| `cross_benzoin_aldehydes_v2.csv.gz` | All 220,859 source aldehydes and metadata |
| `cross_benzoin_dG_candidates_v2.csv.gz` | 2,000,000 directed reactions from 1,000,000 unique unordered pairs |
| `manifest.json` | Counts, split diagnostics, rule version, seed and SHA-256 checksums |
| `cross_benzoin_dG_candidates_v2_QA.xlsx` | Human-readable quality-control summary |

The compressed CSV files are tracked with Git LFS. They can be read directly:

```python
import pandas as pd

reactions = pd.read_csv(
    "data/cross_benzoin/candidates_v2/cross_benzoin_dG_candidates_v2.csv.gz"
)
aldehydes = pd.read_csv(
    "data/cross_benzoin/candidates_v2/cross_benzoin_aldehydes_v2.csv.gz"
)
```

## Design

- All 220,859 input aldehydes are retained: aliphatic, aromatic carbocyclic and
  aromatic heterocyclic.
- No MW or `xtb_risk` exclusion is applied at candidate-generation time.
- Six unordered class combinations are represented approximately equally.
- Every source aldehyde occurs in at least one pair.
- Self-pairs are excluded from this cross set.
- Every unordered pair is expanded into both donor/acceptor orientations because they
  generate different regioisomeric products and can have different free energies.
- The split is molecule-disjoint: 80% train, 10% validation and 10% test by a stable
  SHA-256 hash of InChIKey. Both members of a pair belong to the same split.
- `dG_kcal_mol` is intentionally blank and `dG_status` is `to_be_computed`.

The supplied source library carries `PubChem` as its only provenance label. The set is
structurally broad within that source, but it is not a multi-database provenance set.

## Using the existing featurizer

`cross_benzoin/cb_featurize.py --pairs` expects uncompressed manifests with the columns
`donor_id,acceptor_id,donor_smiles,acceptor_smiles`. Do not decompress and load all two
million rows into a single process. Create bounded manifests instead:

```bash
python cross_benzoin/prepare_pair_chunks.py \
  --input data/cross_benzoin/candidates_v2/cross_benzoin_dG_candidates_v2.csv.gz \
  --out data/cross_benzoin/candidates_v2/chunks_train \
  --split train \
  --chunk-size 5000
```

The generated chunk directories are intentionally ignored by git and can be submitted
as a SLURM array.

## Integrity

- Reaction CSV.gz SHA-256:
  `ce6add948a8f65d06293cd6e8109e9243b2b5bbda1f27f5ede128c6f34aafa86`
- Molecule CSV.gz SHA-256:
  `501315da40d097bcc4ad16d77a0aa95040e86ad41bcf3ebedaeee0777c5d5712`
- Random seed: `20260713`
- Rule version: `cross_benzoin_dG_pairing_v2`

See `cross_benzoin/docs/CROSS_BENZOIN_ML_RECOMMENDATIONS.md` for modeling and
validation recommendations.
