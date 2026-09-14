# Homo aldehyde clustering vs dG_r2scan_kcal -- diagnostic

Generated 2026-09-14T11:15:49.840990+00:00. Requested 2026-09-14: does
chemotype clustering (for the flying-dataset split design, CHEMICAL_SPACE.md
sec8) carry signal for the homo reaction target, or is it orthogonal -- and
does the choice of clustering *method* (KMeans-on-ECFP4 vs the existing
Bemis-Murcko scaffold) and split regime (random vs group-disjoint) change
the answer?

**Caveat: partial labels.** The homo SP relabel campaign is in flight.
94912/220859 aldehydes (43.0%) have a dG_r2scan_kcal
label right now -- re-run this script once the campaign drains for the
full-library numbers; treat these as directional, not final.

## Method
- 220859/220859 aldehydes fingerprinted (ECFP4 r=2, 2048 bit),
  0 SMILES failed to parse and were excluded from clustering.
- MiniBatchKMeans, k=150 (Euclidean on binary bits -- an approximation
  of Tanimoto clustering, not equivalent to it; chosen for O(N*k) scalability
  at full 220,859-molecule scale per 2026-09-14 direction). Cluster sizes:
  min 4, median 1352, max 5395.
- Second grouping method: Bemis-Murcko scaffold (`aldehyde_index.parquet`
  `scaffold` column, already computed for this library by
  `build_scaffold_splits.py`/BDE work) -- 24843
  unique scaffolds in the labeled subset, 118 rows dropped for a
  missing scaffold.
- y=dG_r2scan_kcal (homo self-reaction free energy; homo pairs are
  donor==acceptor so `id` in homo_sp_labels.csv is a single ald_idx -- a
  legitimately per-aldehyde target here, unlike cross's pair-level target).
- Three split regimes x four methods, 5 folds each, same regime-specific
  folds reused across all four methods for a fair within-regime comparison.
  The structure bracket uses ECFP4 folded 2048->128 bits (OR-fold of
  consecutive chunks) rather than a PCA/SVD reduction -- folding is
  data-independent so it needs no per-fold refit and cannot leak test-fold
  structure; switched from an earlier fold-safe-PCA version of this script
  after PCA turned out to need BLAS matmuls this venv's numpy executes
  unvectorized on this node's CPU generation (14s for a single 76000x2048
  matmul that should take <1s), making even one PCA refit minutes long.

## Results (MAE in kcal/mol)

| split regime | method | MAE | R2 |
|---|---|---|---|
| random | global mean (no structure) | 7.974 +/- 0.054 | -0.000 |
| random | group-mean only (cluster or scaffold, per regime) | 7.525 +/- 0.043 | 0.033 |
| random | XGB on group one-hot | 7.651 +/- 0.049 | 0.025 |
| random | XGB on fingerprint folded to 128 bits (structure bracket) | 7.373 +/- 0.046 | 0.053 |
| cluster_disjoint | global mean (no structure) | 7.980 +/- 0.272 | -0.001 |
| cluster_disjoint | group-mean only (cluster or scaffold, per regime) | 7.980 +/- 0.272 | -0.001 |
| cluster_disjoint | XGB on group one-hot | 7.918 +/- 0.267 | -0.001 |
| cluster_disjoint | XGB on fingerprint folded to 128 bits (structure bracket) | 7.509 +/- 0.221 | 0.038 |
| scaffold_disjoint | global mean (no structure) | 7.979 +/- 0.504 | -0.001 |
| scaffold_disjoint | group-mean only (cluster or scaffold, per regime) | 7.979 +/- 0.504 | -0.001 |
| scaffold_disjoint | XGB on group one-hot | 7.979 +/- 0.503 | -0.000 |
| scaffold_disjoint | XGB on fingerprint folded to 128 bits (structure bracket) | 7.455 +/- 0.548 | 0.048 |

(`xgb_group_onehot` / `xgb_fp_folded128` are diagnostic brackets, not
champion-quality models -- no QM descriptors, no baseline subtraction.
Compare their *gap* within a regime, not their absolute MAE, to the champion
pipeline.)

## Reading

**Method-validation (this session's ask):** compare the `cluster_disjoint`
row block to the `scaffold_disjoint` block. If they tell a similar story
(similar MAE/R2 gaps from `random`), the KMeans-on-ECFP4 clustering and the
independently-computed Bemis-Murcko scaffold agree on how hard structural
extrapolation is for this target -- the clustering *method* choice doesn't
matter much here, and there's no need to spend the ~20-40 min for an exact
Butina/Tanimoto clustering just to validate it further. If they diverge, the
method choice does matter and Butina is the next step.

**Split-difficulty:** compare `random` to the two `*_disjoint` regimes. A
much worse `group_mean`/`xgb_group_onehot` MAE under `*_disjoint` (like the
scaffold-disjoint correction found for BDE/cross,
[[bde-scaffold-leakage-finding]], [[cross-r1-10-champion]]) means a
structurally-grouped flying-dataset split is the honest one to build, and a
random split would overstate how well any 2D-structure-only model
generalizes. A small gap instead means 2D structure just doesn't carry much
signal for `dG_r2scan_kcal` at all (compare all four `xgb_fp_folded128` R2
values to the near-zero `global_mean` R2) -- consistent with
[[homo_active_relabel_null_result]]'s redirect: the signal this target needs
is in QM/electronic descriptors the champion pipeline computes, not in gross
2D chemotype similarity.
