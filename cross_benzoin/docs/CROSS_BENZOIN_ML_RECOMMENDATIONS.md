# Cross-benzoin ML extension: recommendations and lessons from the homo model

## Executive recommendation

The homo-benzoin model is a valuable pretraining task, but it samples only the diagonal
of the pair space: `homo(A) = cross(A, A)`. It teaches single-substrate electronic and
thermochemical corrections; it does not by itself teach donor/acceptor complementarity,
regioisomer choice or out-of-diagonal pair interactions.

The recommended cross workflow is therefore:

1. retain the validated GFN2 geometry/frequency plus g-xTB/DFT single-point composite;
2. reuse the homo molecular encoders and descriptor pipeline;
3. fine-tune on directed cross products with explicit donor, acceptor and product roles;
4. learn the high-level correction, not the high-level free energy from scratch;
5. evaluate with pair-, molecule- and scaffold-disjoint protocols.

## Interpreting the 40% GNN + 60% tabular ensemble

An optimized 40/60 prediction weight is not a 40/60 feature-importance statement. It
only describes the best linear blend on the tuning data. Before promoting the reported
MAE of 1.427 kcal/mol, retain the following evidence in the repository:

- GNN-only, tabular-only, 50/50 and tuned-blend metrics on exactly the same test rows;
- the Pearson and Spearman correlation between the two models' residuals;
- the split-generation code, fixed row IDs and random seeds;
- out-of-fold predictions used to fit the blend weight;
- MAE, RMSE, median AE, P90/P95 AE, mean signed error and per-class MAE;
- uncertainty coverage and error-versus-uncertainty curves.

If the weight was selected on the final test set, the 1.427 result is exploratory and a
new untouched test set is required. A stronger production approach is out-of-fold
stacking: train a small Ridge or constrained linear meta-model on OOF GNN and tabular
predictions. A class- or uncertainty-dependent gating model is worth testing only after
the fixed linear stack is stable.

## Thermodynamically structured delta learning

For species `M`, define the single-point correction on the same GFN2 geometry:

```text
delta_E(M) = E_high_SP(M) - E_g-xTB_SP(M)
```

Then enforce the reaction cycle:

```text
delta_delta_G = delta_E(P) - delta_E(D) - delta_E(A)
G_high_pred = G_g-xTB_reaction + delta_delta_G
```

The most reusable architecture is a shared species encoder that predicts `delta_E(M)`
for donor, acceptor and product, followed by the exact stoichiometric combination above.
For homo-benzoin this automatically reduces to `delta_E(P) - 2*delta_E(A)`. An
atom-wise additive readout is preferable when predicting total-energy corrections.

A direct reaction-residual model remains a useful benchmark, especially if systematic
errors are not perfectly species-additive, but it should not be the only model.

## Recommended cross representation

Use three role-aware molecular inputs:

```text
h_D = Encoder(donor) + donor_role
h_A = Encoder(acceptor) + acceptor_role
h_P = Encoder(product) + product_role

h_rxn = [h_D, h_A, h_P, h_P-h_D-h_A, abs(h_D-h_A), h_D*h_A]
```

The product is essential because opposite donor/acceptor directions generate different
regioisomers. If inference starts with only two aldehyde SMILES, product generation must
be deterministic, atom-mapped and validated before featurization.

Use homo samples for encoder pretraining and as a diagonal consistency constraint:

```text
L_consistency = abs(f_cross(A, A) - f_homo(A))
```

Mix homo and cross batches during fine-tuning so that the cross extension does not
catastrophically forget the well-covered diagonal chemistry.

## Descriptor policy

The GNN does not require handcrafted descriptors, but the current homo results show that
the low-level physics layer contains most of the useful information. Retain three groups:

1. **2D structure:** molecular graph, Morgan counts, MW, TPSA, LogP, HBD/HBA, rings,
   rotatable bonds and formal charge.
2. **Low-level quantum:** g-xTB/GFN2 energy, thermal correction, solvation energy,
   HOMO/LUMO/gap, dipole, polarizability, reaction-center charges, Wiberg bond orders and
   Fukui/electrophilicity terms.
3. **Conformer/QC:** conformer count, energy span, lowest frequency, imaginary-frequency
   count, optimization status and calculation-failure flags.

Store donor, acceptor and product values separately and add the stoichiometric difference
`feature(P) - feature(D) - feature(A)`. Do not use DFT-derived descriptors as inputs to a
model whose purpose is to avoid DFT at inference time.

The operational keep/add/prune decisions and donor-versus-acceptor responsibilities are
specified in [DESCRIPTOR_POLICY_CROSS.md](DESCRIPTOR_POLICY_CROSS.md).

## Product-first computation and aldehyde reuse

The aldehyde simulations should not be repeated for every pair. Canonicalize the two
SMILES, join their stored GFN2/g-xTB free energies and descriptors, and compute only
missing species. Before conformer or QM work, generate the directed product with
`prepare_product_manifest.py`; reject invalid products and preserve the orientation ID.
`cb_featurize.py --aldehyde-cache ... --require-cache-complete` then guarantees that a
product batch cannot silently launch expensive aldehyde recalculations.

## Splits needed for cross-benzoin

One aggregate random-split MAE is insufficient. Keep both orientations of an unordered
pair in the same split and report at least four scenarios:

| Evaluation | Meaning |
|---|---|
| Homo diagonal | Compatibility with the established `A == B` task |
| New pair, known molecules | Tests whether the model learns combinations |
| One unseen molecule | Single-sided molecular extrapolation |
| Both unseen, scaffold-disjoint | True chemical-space extrapolation |

Also stratify by the six class pairs, molecular size/flexibility, functional-group risk
and computation-success status. The v2 candidate split is molecule-disjoint; a second
scaffold-family-disjoint benchmark should be generated after RDKit standardization.

## Computation and active-learning strategy

Do not calculate all two million directed candidates at DFT level. A practical loop is:

1. start with a diverse, class-balanced and functional-group-aware subset;
2. compute the existing GFN2/g-xTB/DFT labels and QC metadata;
3. train an ensemble with calibrated uncertainty;
4. select high-uncertainty plus high-diversity candidates;
5. repeat until the external/scaffold-disjoint learning curve plateaus.

Candidate selection should operate on unordered pairs, then compute both orientations.
This avoids spending the budget on only one regioisomer and preserves paired analysis.

## Repository engineering priorities

1. Stream or chunk the two-million-row manifest. `cb_featurize.py` currently materializes
   the complete input and submits one future per pair; that is unsuitable at this scale.
2. Record `method_id`, solvent, temperature, pressure/standard state, charge,
   multiplicity and geometry provenance in every result row.
3. Make g-xTB failures explicit (`gxtb_sp_failed`, `gxtb_dG_failed`) instead of silently
   leaving a nullable value.
4. Correct the `g_cache` annotation: values are `(G_gfn2, G_gxtb)` tuples, not scalars.
5. Keep `README.md`, `ARCHITECTURE.md` and `FILE_MAP.md` synchronized with the current
   220k-scale production results and the final leak-free stacking benchmark.
6. Publish large candidate data through Git LFS or a versioned data release with SHA-256
   checksums; do not commit generated geometries or per-task scratch.

## Suggested promotion gate for the cross model

A cross model should become production only when it:

- beats the g-xTB-only and tabular-only baselines on the untouched molecule-disjoint set;
- shows a reproducible stacking gain from OOF predictions;
- maintains acceptable homo-diagonal performance;
- has calibrated uncertainty that enriches actual large errors;
- reports failure rates and coverage, not only MAE on successful calculations;
- passes a small independently recomputed set with frozen methods and conformer settings.
