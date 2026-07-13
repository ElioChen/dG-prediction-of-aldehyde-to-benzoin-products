# Legacy / superseded scripts

Kept for provenance; **do not extend**. Superseded by the unified r2SCAN-3c path
(see ../../ARCHITECTURE.md).

- `select_subset.py` — PCA+KMeans subset selection → use `expand_subset.py` (MaxMin).
- `merge_labels.py`, `add_expand_chunk.py` — chunk-CSV label assembly → use
  `assemble_featurize.py` (one parquet).
- `run_/submit_descriptors_array.sh`, `run_/submit_labels_array.sh` — separate
  descriptors + ORCA(PBE0-D4) path → use `run_/submit_featurize_array.sh` (unified,
  r2SCAN-3c, geometry-consistent).
