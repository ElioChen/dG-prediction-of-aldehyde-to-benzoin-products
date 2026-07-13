# Cross-benzoin implementation roadmap

The target product is one predictor with a directed input `(donor SMILES, acceptor
SMILES)`. Homo-benzoin is the diagonal case `(A, A)`, not a separate chemistry mode.
The current shipped model remains homo-only until the cross acceptance gates below pass.

## Phase 0 — candidate space (complete)

- Publish v3: 220,859 aldehydes, 2,000,000 unique non-self pairs and 4,000,000 directed rows.
- Preserve aliphatic, aromatic-carbo and aromatic-hetero coverage.
- Require molecule-disjoint 80/10/10 splits and zero AB/BA pairing errors.

## Phase 1 — product enumeration and QC (implemented; execute in chunks)

- Run `prepare_pair_chunks.py`, then `prepare_product_manifest.py`.
- Preserve both orientations and reject unsanitized/non-benzoin products before QM.
- Atom-map a validation subset and visually inspect representatives of all six class pairs.
- Acceptance: at least 99% valid products or a documented chemistry-specific reject rule.

## Phase 2 — label acquisition

- Reuse the existing aldehyde GFN2/g-xTB/descriptor table by canonical SMILES.
- Calculate only new products: conformer funnel → GFN2 optimization/frequency → g-xTB
  single point; use the existing DFT single-point correction workflow on selected rows.
- Do **not** run DFT over all 4M candidates. Start with a molecule-/class-stratified
  diversity set, then add disagreement/uncertainty active-learning rounds.
- Acceptance: connectivity preserved, frequency/QC status stored, failures never hidden,
  and reaction ΔG reconstructed from species energies with a method/version identifier.

## Phase 3 — cross training table and baselines

- Assemble `donor_*`, `acceptor_*`, `product_*`, `delta_*` and interaction blocks using
  `DESCRIPTOR_POLICY_CROSS.md`.
- Train in this order: g-xTB baseline; tabular Δ-model; directed GNN; fixed 40/60 blend;
  OOF stacking; optional uncertainty-aware gating.
- Pretrain the species encoder on homo, mix homo/cross during fine-tuning, and enforce
  `f_cross(A,A) ≈ f_homo(A)` as a diagonal consistency test.

## Phase 4 — evaluation and promotion

- Freeze random, molecule-disjoint and scaffold-/cluster-disjoint test sets before tuning.
- Report MAE, RMSE, median AE, P90/P95 AE, signed error, class-pair/orientation slices,
  calibration and inference failure rate.
- The reported GNN(40%)+tabular(60%) MAE 1.427 is a candidate result until its weight is
  fitted from OOF predictions and reproduced on an untouched split.
- Promote cross inference only when it beats g-xTB and both standalone models on the
  untouched test, with no hidden product/QM failures.

## Phase 5 — package/API

- Add `predict_cross_dG(donor_smiles, acceptor_smiles)` and a pair-aware CLI mode.
- Route `(A,A)` through the same API and compare with the frozen homo predictor.
- Ship model metadata: chemistry scope, label method, dataset version, split hash,
  feature schema, uncertainty calibration and applicability-domain reference.
